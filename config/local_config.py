"""
Configuration file for AVAE disentanglement evaluation.

Update these settings based on your trained model checkpoints and desired evaluation parameters.
"""

# Configuration dictionary
# Key structure: {config_id: {parameter_dict}}
configurations = {
    0: {
        # Model identification
        'model_name': "AVAE",
        'dataset_name': "DSprites",
        
        # Model architecture
        'latent_dim': 6,              # Latent dimension (set to 6 for known factors in DSprites/3DShapes)
        'num_filter': 64,             # Number of filters in conv layers
        
        # Data parameters
        'batch_size': 100,
        'train_data_size': 10000,
        'num_train_data': 10000,      # Number of training samples for metric computation
        'num_eval_data': 5000,        # Number of evaluation samples
        'samples_for_global_var': 10000,  # Samples for computing whitening matrix
        
        # Evaluation parameters
        'use_whiten_data': False,     # Whether to whiten latent representations
        'encoder_use_batch_norm': True,
        'num_bins': 10,               # Number of bins for discretization
        
        # Model checkpoint path
        # Update this to point to your trained model directory
        # Option 1: If models are in AVAE training repo
        'model_checkpoint_dir': '../../../AVAE/AVAE/logs/DSprites/Run_{run_id}/Models/best_model',
        # Option 2: If you copied models to Research/logs
        # 'model_checkpoint_dir': '../../logs/DSprites/Run_{run_id}/Models/best_model',
        
        # Output directory
        'output_dir': 'Output/DSprites',
    },
    
    1: {
        # Model identification
        'model_name': "AVAE",
        'dataset_name': "Shapes3D",
        
        # Model architecture
        'latent_dim': 6,              # Latent dimension
        'num_filter': 64,             # Number of filters
        
        # Data parameters
        'batch_size': 100,
        'train_data_size': 10000,
        'num_train_data': 10000,
        'num_eval_data': 5000,
        'samples_for_global_var': 10000,
        
        # Evaluation parameters
        'use_whiten_data': False,
        'encoder_use_batch_norm': True,
        'num_bins': 10,
        
        # Model checkpoint path
        # Option 1: If models are in AVAE training repo
        'model_checkpoint_dir': '../../../AVAE/AVAE/logs/Shapes3D/Run_{run_id}/Models/best_model',
        # Option 2: If you copied models to Research/logs
        # 'model_checkpoint_dir': '../../logs/Shapes3D/Run_{run_id}/Models/best_model',
        
        # Output directory
        'output_dir': 'Output/Shapes3D',
    },
}

# Run IDs for multiple independent evaluations (as in paper: 10 runs)
EVAL_RUN_IDS = list(range(1, 11))  # [1, 2, 3, ..., 10]

# Dataset-specific configurations
DATASET_INFO = {
    'DSprites': {
        'image_shape': (64, 64, 1),
        'num_factors': 6,  # Color, Shape, Scale, Orientation, PositionX, PositionY
        'factor_names': ['Color', 'Shape', 'Scale', 'Orientation', 'PositionX', 'PositionY'],
        'factor_sizes': [1, 3, 6, 40, 32, 32],  # From dsprites metadata
    },
    'Shapes3D': {
        'image_shape': (64, 64, 3),
        'num_factors': 6,  # Floor hue, Wall hue, Object hue, Scale, Shape, Orientation
        'factor_names': ['Floor Hue', 'Wall Hue', 'Object Hue', 'Scale', 'Shape', 'Orientation'],
        'factor_sizes': [10, 10, 10, 8, 4, 15],
    },
}
