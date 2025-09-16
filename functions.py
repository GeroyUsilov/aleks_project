"""
ProstT5 Translation Functions
Simple translation functions for AA<->3Di conversion using ProstT5.
"""

from transformers import T5Tokenizer, AutoModelForSeq2SeqLM
import torch
import re
from typing import List, Union, Optional, Dict, Any

# Global model cache to avoid reloading
_model_cache = {}

def _load_model(model_name: str = "Rostlab/ProstT5", device: Optional[str] = None):
    """Load and cache the ProstT5 model and tokenizer."""
    global _model_cache
    
    if device is None:
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    cache_key = f"{model_name}_{device}"
    
    if cache_key not in _model_cache:
        device_obj = torch.device(device)
        
        tokenizer = T5Tokenizer.from_pretrained(model_name, do_lower_case=False)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device_obj)
        
        # Use half precision for GPU, full precision for CPU
        if device_obj.type == 'cuda':
            model = model.half()
        else:
            model = model.float()
        
        _model_cache[cache_key] = {
            'tokenizer': tokenizer,
            'model': model,
            'device': device_obj
        }
    
    return _model_cache[cache_key]

def _preprocess_sequences(sequences: List[str], translation_type: str) -> List[str]:
    """Preprocess sequences for translation."""
    processed_sequences = []
    
    for sequence in sequences:
        # Replace rare/ambiguous amino acids with X and add spaces
        processed_seq = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
        
        # Add appropriate prefix
        if translation_type == "aa2fold":
            processed_seq = "<AA2fold> " + processed_seq
        elif translation_type == "fold2aa":
            processed_seq = "<fold2AA> " + processed_seq
        else:
            raise ValueError("translation_type must be 'aa2fold' or 'fold2aa'")
        
        processed_sequences.append(processed_seq)
    
    return processed_sequences

def translate_aa_to_3di(sequences: Union[str, List[str]], 
                       device: Optional[str] = None,
                       **generation_kwargs) -> List[str]:
    """
    Translate amino acid sequences to 3Di structural sequences.
    
    Args:
        sequences: Single sequence string or list of sequences
        device: Device to use ('cuda:0', 'cpu', etc.). Auto-detects if None.
        **generation_kwargs: Additional generation parameters
        
    Returns:
        List of 3Di sequences (lowercase)
    """
    if isinstance(sequences, str):
        sequences = [sequences]
    
    # Load model components
    components = _load_model(device=device)
    tokenizer = components['tokenizer']
    model = components['model']
    device_obj = components['device']
    
    # Default generation parameters for AA->3Di
    default_gen_kwargs = {
        "do_sample": True,
        "num_beams": 3,
        "top_p": 0.95,
        "temperature": 1.2,
        "top_k": 6,
        "repetition_penalty": 1.2,
    }
    default_gen_kwargs.update(generation_kwargs)
    
    # Preprocess sequences
    processed_sequences = _preprocess_sequences(sequences, "aa2fold")
    
    # Calculate lengths for generation
    min_len = min([len(s.replace(" ", "")) for s in processed_sequences])
    max_len = max([len(s.replace(" ", "")) for s in processed_sequences])
    
    # Tokenize
    ids = tokenizer.batch_encode_plus(
        processed_sequences,
        add_special_tokens=True,
        padding="longest",
        return_tensors='pt'
    ).to(device_obj)
    
    # Generate translations
    with torch.no_grad():
        translations = model.generate(
            ids.input_ids,
            attention_mask=ids.attention_mask,
            max_length=max_len,
            min_length=min_len,
            early_stopping=True,
            num_return_sequences=1,
            **default_gen_kwargs
        )
    
    # Decode and clean up
    decoded_translations = tokenizer.batch_decode(translations, skip_special_tokens=True)
    structure_sequences = ["".join(ts.split(" ")) for ts in decoded_translations]
    
    return structure_sequences

# Cache for 3Di->AA generation kwargs
_fold2aa_default_kwargs = {
    "do_sample": True,
    "top_p": 0.85,
    "temperature": 1.0,
    "top_k": 3,
    "repetition_penalty": 1.2,
}

def translate_3di_to_aa(sequences: Union[str, List[str]], 
                       device: Optional[str] = None,
                       **generation_kwargs) -> List[str]:
    """
    Translate 3Di structural sequences to amino acid sequences.
    
    Args:
        sequences: Single 3Di sequence string or list of sequences
        device: Device to use ('cuda:0', 'cpu', etc.). Auto-detects if None.
        **generation_kwargs: Additional generation parameters
        
    Returns:
        List of amino acid sequences (uppercase)
    """
    if isinstance(sequences, str):
        sequences = [sequences]
    
    # Load model components (cached after first call)
    components = _load_model(device=device)
    tokenizer = components['tokenizer']
    model = components['model']
    device_obj = components['device']
    
    # Use cached default kwargs and update if needed
    if generation_kwargs:
        gen_kwargs = {**_fold2aa_default_kwargs, **generation_kwargs}
    else:
        gen_kwargs = _fold2aa_default_kwargs
    
    # Preprocess sequences (add spaces between characters)
    processed_sequences = ["<fold2AA> " + " ".join(seq) for seq in sequences]
    
    # Calculate lengths for generation
    seq_lengths = [len(s) for s in sequences]
    min_len = min(seq_lengths)
    max_len = max(seq_lengths)
    
    # Tokenize
    ids = tokenizer.batch_encode_plus(
        processed_sequences,
        add_special_tokens=True,
        padding="longest",
        return_tensors='pt'
    ).to(device_obj)
    
    # Generate translations
    with torch.no_grad():
        translations = model.generate(
            ids.input_ids,
            attention_mask=ids.attention_mask,
            max_length=max_len,
            min_length=min_len,
            num_return_sequences=1,
            **gen_kwargs
        )
    
    # Decode and clean up (more efficient string operations)
    decoded_translations = tokenizer.batch_decode(translations, skip_special_tokens=True)
    amino_acid_sequences = [ts.replace(" ", "") for ts in decoded_translations]
    
    return amino_acid_sequences

def clear_model_cache():
    """Clear the model cache to free GPU memory."""
    global _model_cache
    _model_cache.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()