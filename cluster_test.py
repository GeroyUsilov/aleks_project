from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import torch
import re
import os
import time

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Path to local model directory
model_path = os.path.join(os.getcwd(), '../models')

print(f"Loading model from: {model_path}")
print(f"Using device: {device}")
print("="*50)

# Timer: Model loading
start_time = time.time()
print("Loading tokenizer...")
tokenizer_start = time.time()

# Load the tokenizer from local directory
tokenizer = T5Tokenizer.from_pretrained(model_path, do_lower_case=False)

tokenizer_time = time.time() - tokenizer_start
print(f"✓ Tokenizer loaded in {tokenizer_time:.2f}s")

print("Loading model...")
model_start = time.time()

# Load the model from local directory
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

model_load_time = time.time() - model_start
print(f"✓ Model loaded in {model_load_time:.2f}s")

print("Converting model precision...")
precision_start = time.time()

# only GPUs support half-precision currently; if you want to run on CPU use full-precision (not recommended, much slower)
model.float() if device.type=='cpu' else model.half()

precision_time = time.time() - precision_start
total_setup_time = time.time() - start_time
print(f"✓ Model precision converted in {precision_time:.2f}s")
print(f"✓ Total setup time: {total_setup_time:.2f}s")
print("="*50)

# prepare your protein sequences/structures as a list.
# Amino acid sequences are expected to be upper-case ("PRTEINO" below)
# while 3Di-sequences need to be lower-case.
sequence_examples = ["PRTEINO", "SEQWENCE"]
min_len = min([ len(s) for s in sequence_examples])
max_len = max([ len(s) for s in sequence_examples])

print("Data preparation...")
prep_start = time.time()

# replace all rare/ambiguous amino acids by X (3Di sequences does not have those) and introduce white-space between all sequences (AAs and 3Di)
sequence_examples = [" ".join(list(re.sub(r"[UZOB]", "X", sequence))) for sequence in sequence_examples]

# add pre-fixes accordingly. For the translation from AAs to 3Di, you need to prepend "<AA2fold>"
sequence_examples = [ "<AA2fold>" + " " + s for s in sequence_examples]

print(f"Processing sequences: {sequence_examples}")

# tokenize sequences and pad up to the longest sequence in the batch
ids = tokenizer.batch_encode_plus(sequence_examples,
                                  add_special_tokens=True,
                                  padding="longest",
                                  return_tensors='pt').to(device)

prep_time = time.time() - prep_start
print(f"✓ Data preparation completed in {prep_time:.3f}s")

# Generation configuration for "folding" (AA-->3Di)
gen_kwargs_aa2fold = {
                  "do_sample": True,
                  "num_beams": 3, 
                  "top_p" : 0.95, 
                  "temperature" : 1.2, 
                  "top_k" : 6,
                  "repetition_penalty" : 1.2,
}

print("Translating AA to 3Di...")
# translate from AA to 3Di (AA-->3Di)
with torch.no_grad():
      translations = model.generate( 
              ids.input_ids, 
              attention_mask=ids.attention_mask, 
              max_length=max_len, # max length of generated text
              min_length=min_len, # minimum length of the generated text
              early_stopping=True, # stop early if end-of-text token is generated
              num_return_sequences=1, # return only a single sequence
              **gen_kwargs_aa2fold
  )

# Decode and remove white-spaces between tokens
decoded_translations = tokenizer.batch_decode( translations, skip_special_tokens=True )
structure_sequences = [ "".join(ts.split(" ")) for ts in decoded_translations ] # predicted 3Di strings

print(f"3Di sequences: {structure_sequences}")

print("\n" + "="*50)
print("3Di → AA BACK-TRANSLATION")
print("="*50)

# Timer: Data preparation for back-translation
backtrans_prep_start = time.time()

# Now we can use the same model and invert the translation logic
# to generate an amino acid sequence from the predicted 3Di-sequence (3Di-->AA)

# add pre-fixes accordingly. For the translation from 3Di to AA (3Di-->AA), you need to prepend "<fold2AA>"
sequence_examples_backtranslation = [ "<fold2AA>" + " " + s for s in decoded_translations]

# tokenize sequences and pad up to the longest sequence in the batch
ids_backtranslation = tokenizer.batch_encode_plus(sequence_examples_backtranslation,
                                  add_special_tokens=True,
                                  padding="longest",
                                  return_tensors='pt').to(device)

backtrans_prep_time = time.time() - backtrans_prep_start
print(f"✓ Back-translation data prep: {backtrans_prep_time:.3f}s")

# Example generation configuration for "inverse folding" (3Di-->AA)
gen_kwargs_fold2AA = {
            "do_sample": True,
            "top_p" : 0.85,
            "temperature" : 1.0,
            "top_k" : 3,
            "repetition_penalty" : 1.2,
}

# Timer: 3Di to AA translation
fold2aa_start = time.time()
print("Translating 3Di back to AA...")

# translate from 3Di to AA (3Di-->AA)
with torch.no_grad():
      backtranslations = model.generate( 
              ids_backtranslation.input_ids, 
              attention_mask=ids_backtranslation.attention_mask, 
              max_length=max_len, # max length of generated text
              min_length=min_len, # minimum length of the generated text
              #early_stopping=True, # stop early if end-of-text token is generated; only needed for beam-search
              num_return_sequences=1, # return only a single sequence
              **gen_kwargs_fold2AA
)

fold2aa_generation_time = time.time() - fold2aa_start

# Timer: Final decoding
final_decode_start = time.time()
# Decode and remove white-spaces between tokens
decoded_backtranslations = tokenizer.batch_decode( backtranslations, skip_special_tokens=True )
aminoAcid_sequences = [ "".join(ts.split(" ")) for ts in decoded_backtranslations ]
final_decode_time = time.time() - final_decode_start

total_fold2aa_time = time.time() - fold2aa_start
print(f"✓ 3Di→AA generation: {fold2aa_generation_time:.2f}s")
print(f"✓ Final decoding: {final_decode_time:.3f}s")
print(f"✓ Total 3Di→AA time: {total_fold2aa_time:.2f}s")

print("\n" + "="*50)
print("FINAL RESULTS")
print("="*50)
print(f"Original AA sequences: {['PRTEINO', 'SEQWENCE']}")
print(f"Predicted 3Di sequences: {structure_sequences}")
print(f"Back-translated AA sequences: {aminoAcid_sequences}")

# Calculate and display total runtime
total_runtime = time.time() - start_time
print("\n" + "="*50)
print("TIMING SUMMARY")
print("="*50)
print(f"Setup time:           {total_setup_time:.2f}s")
print(f"  - Tokenizer:        {tokenizer_time:.2f}s")
print(f"  - Model loading:    {model_load_time:.2f}s")
print(f"  - Precision conv:   {precision_time:.2f}s")
print(f"Data preparation:     {prep_time:.3f}s")
print(f"AA→3Di translation:   {total_aa2fold_time:.2f}s")
print(f"  - Generation:       {aa2fold_generation_time:.2f}s")
print(f"  - Decoding:         {decode_time:.3f}s")
print(f"Back-translation:     {total_fold2aa_time:.2f}s")
print(f"  - Data prep:        {backtrans_prep_time:.3f}s")
print(f"  - Generation:       {fold2aa_generation_time:.2f}s")
print(f"  - Decoding:         {final_decode_time:.3f}s")
print(f"TOTAL RUNTIME:        {total_runtime:.2f}s")
print("="*50)