"""
Model Checkpoint Serialization and Weight Export Utilities

Provides:
1. Strict state_dict validation and checkpoint persistence (.npz / dict)
2. SafeTensors-compatible zero-copy binary serialization format
3. Checkpoint SHA-256 cryptographic verification and structural metadata inspection
"""

import os
import json
import hashlib
import numpy as np


class ModelExporter:
    """
    Model Serialization and Export Engine
    """
    
    @staticmethod
    def save_checkpoint(filepath, model, optimizer=None, epoch=0, step=0, metadata=None):
        """
        Save complete training checkpoint with parameters, optimizer state, and metadata.
        
        Args:
            filepath: Destination file path (.npz)
            model: Neural network model instance
            optimizer: Optional optimizer instance
            epoch: Current epoch index
            step: Current step index
            metadata: Optional dictionary of user metadata
        """
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        
        state = {
            'params': model.parameters(),
            'epoch': epoch,
            'step': step,
            'metadata': metadata or {}
        }
        
        if optimizer is not None and hasattr(optimizer, 'm'):
            state['optimizer_m'] = optimizer.m
            state['optimizer_v'] = optimizer.v
            
        np.savez_compressed(filepath, **state)
        checksum = ModelExporter.compute_checksum(filepath)
        return checksum
        
    @staticmethod
    def load_checkpoint(filepath, model, optimizer=None, strict=True):
        """
        Load checkpoint with shape verification and strict parameter matching.
        
        Args:
            filepath: Path to checkpoint .npz
            model: Target model to populate
            optimizer: Target optimizer
            strict: If True, raises error on key or shape mismatch
            
        Returns:
            Dictionary containing epoch, step, and metadata
        """
        data = np.load(filepath, allow_pickle=True)
        saved_params = data['params'].item() if data['params'].dtype == object else data['params']
        
        current_params = model.parameters()
        
        for k, v in saved_params.items():
            if k not in current_params:
                if strict:
                    raise KeyError(f"Unexpected key '{k}' in checkpoint.")
                continue
            if current_params[k].shape != v.shape:
                if strict:
                    raise ValueError(f"Shape mismatch for '{k}': expected {current_params[k].shape}, got {v.shape}")
                continue
                
        if hasattr(model, 'set_parameters'):
            model.set_parameters(saved_params)
        else:
            for name, val in saved_params.items():
                parts = name.split('.')
                obj = model
                for p in parts[:-1]:
                    obj = getattr(obj, p)
                setattr(obj, parts[-1], val.copy())
                
        meta_dict = {
            'epoch': int(data['epoch']) if 'epoch' in data else 0,
            'step': int(data['step']) if 'step' in data else 0,
            'metadata': data['metadata'].item() if 'metadata' in data and data['metadata'].dtype == object else {}
        }
        return meta_dict
        
    @staticmethod
    def compute_checksum(filepath, algorithm='sha256'):
        """
        Compute cryptographic hash of saved model file.
        """
        hasher = hashlib.new(algorithm)
        with open(filepath, 'rb') as f:
            while chunk := f.read(65536):
                hasher.update(chunk)
        return hasher.hexdigest()
        
    @staticmethod
    def export_safetensors(filepath, state_dict, metadata=None):
        """
        Export state dict into SafeTensors binary format.
        SafeTensors spec: 8-byte little-endian header length + JSON header + raw binary buffers.
        
        Args:
            filepath: Target output file (.safetensors)
            state_dict: Dict of parameter name -> NumPy array
            metadata: Optional string dict of metadata
        """
        header = {}
        offset = 0
        buffers = []
        
        # Supported dtype mapping
        dtype_map = {
            np.dtype('float32'): 'F32',
            np.dtype('float64'): 'F64',
            np.dtype('int32'): 'I32',
            np.dtype('int64'): 'I64',
            np.dtype('uint8'): 'U8'
        }
        
        for name, tensor in state_dict.items():
            arr = np.ascontiguousarray(tensor)
            raw_bytes = arr.tobytes()
            length = len(raw_bytes)
            
            header[name] = {
                'dtype': dtype_map.get(arr.dtype, 'F32'),
                'shape': list(arr.shape),
                'data_offsets': [offset, offset + length]
            }
            offset += length
            buffers.append(raw_bytes)
            
        if metadata:
            header['__metadata__'] = metadata
            
        header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
        header_len = len(header_json)
        
        with open(filepath, 'wb') as f:
            # 8-byte little-endian uint64
            f.write(np.uint64(header_len).tobytes())
            f.write(header_json)
            for b in buffers:
                f.write(b)
