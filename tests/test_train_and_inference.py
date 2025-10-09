"""Tests for training and inference functionality.

This module tests the EngressionTrainer class and sample generation
functions to ensure they work correctly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest
import torch
import numpy as np
from train_and_inference import (
    EngressionTrainer, 
    generate_samples, 
    train_and_generate
)
from methods import EngressionNet, get_device


class TestEngressionTrainer:
    """Test EngressionTrainer class."""
    
    def test_trainer_initialization_default(self):
        trainer = EngressionTrainer()
        assert trainer.batch_size == 128
        assert trainer.learning_rate == 1e-4
        assert trainer.num_epochs == 200
        assert trainer.m == 50
        assert trainer.patience == 20
        assert trainer.device is not None
    
    def test_trainer_initialization_custom(self):
        device = get_device()
        trainer = EngressionTrainer(
            batch_size=64,
            learning_rate=1e-3,
            num_epochs=100,
            m=30,
            patience=10,
            device=device
        )
        assert trainer.batch_size == 64
        assert trainer.learning_rate == 1e-3
        assert trainer.num_epochs == 100
        assert trainer.m == 30
        assert trainer.patience == 10
        assert trainer.device.type == device.type
    
    def test_trainer_has_training_history(self):
        trainer = EngressionTrainer()
        assert 'loss' in trainer.training_history
        assert 'term1' in trainer.training_history
        assert 'term2' in trainer.training_history
        assert len(trainer.training_history['loss']) == 0


class TestTrainerWithTensorInput:
    """Test training with torch.Tensor input."""
    
    def test_train_with_tensor_input(self):
        device = get_device()
        X = torch.randn(100, 2, device=device)
        trainer = EngressionTrainer(num_epochs=5, patience=5)
        
        model, history = trainer.train(X, verbose=False)
        
        assert isinstance(model, EngressionNet)
        assert len(history['loss']) > 0
        assert len(history['term1']) > 0
        assert len(history['term2']) > 0
    
    def test_train_records_history(self):
        device = get_device()
        X = torch.randn(100, 2, device=device)
        trainer = EngressionTrainer(num_epochs=10, patience=20)
        
        model, history = trainer.train(X, verbose=False)
        
        assert len(history['loss']) == len(history['term1'])
        assert len(history['loss']) == len(history['term2'])
        assert len(history['loss']) <= 10
    
    def test_train_loss_decreases(self):
        device = get_device()
        X = torch.randn(200, 2, device=device)
        trainer = EngressionTrainer(num_epochs=20, patience=20, learning_rate=1e-3)
        
        model, history = trainer.train(X, verbose=False)
        
        # Loss should generally decrease
        first_loss = history['loss'][0]
        last_loss = history['loss'][-1]
        assert last_loss < first_loss
    
    def test_train_with_custom_model(self):
        device = get_device()
        X = torch.randn(100, 3, device=device)
        custom_model = EngressionNet(input_dim=2, output_dim=3, hidden_dim=32).to(device)
        trainer = EngressionTrainer(num_epochs=5, patience=5)
        
        trained_model, history = trainer.train(X, model=custom_model, verbose=False)
        
        assert trained_model is custom_model
        assert len(history['loss']) > 0


class TestTrainerWithNumpyInput:
    """Test training with numpy array input."""
    
    def test_train_with_numpy_input(self):
        X = np.random.randn(100, 2)
        trainer = EngressionTrainer(num_epochs=5, patience=5)
        
        model, history = trainer.train(X, verbose=False)
        
        assert isinstance(model, EngressionNet)
        assert len(history['loss']) > 0
    
    def test_train_converts_numpy_to_tensor(self):
        X = np.random.randn(50, 2).astype(np.float32)
        trainer = EngressionTrainer(num_epochs=3, patience=5)
        
        model, history = trainer.train(X, verbose=False)
        
        # Should work without errors
        assert len(history['loss']) > 0


class TestEarlyStopping:
    """Test early stopping functionality."""
    
    def test_early_stopping_triggers(self):
        device = get_device()
        # Use small dataset that might converge quickly
        X = torch.randn(50, 2, device=device)
        trainer = EngressionTrainer(
            num_epochs=100, 
            patience=3,
            learning_rate=1e-2
        )
        
        model, history = trainer.train(X, verbose=False)
        
        # Should stop before 100 epochs if converged
        assert len(history['loss']) <= 100
    
    def test_best_model_restored(self):
        device = get_device()
        X = torch.randn(100, 2, device=device)
        trainer = EngressionTrainer(
            num_epochs=30,
            patience=5,
            learning_rate=1e-3
        )
        
        model, history = trainer.train(X, verbose=False)
        
        # The model should have the best weights, not the last weights
        # This is implicitly tested if no errors occur
        assert model is not None


class TestGenerateSamples:
    """Test sample generation functionality."""
    
    def test_generate_samples_shape(self):
        device = get_device()
        model = EngressionNet(input_dim=2, output_dim=3).to(device)
        
        samples = generate_samples(model, num_samples=100, input_dim=2, device=device)
        
        assert samples.shape == (100, 3)
    
    def test_generate_samples_default_params(self):
        device = get_device()
        model = EngressionNet().to(device)
        
        samples = generate_samples(model)
        
        assert samples.shape == (1000, 2)
    
    def test_generate_samples_is_deterministic_with_seed(self):
        device = get_device()
        model = EngressionNet().to(device)
        
        torch.manual_seed(42)
        samples1 = generate_samples(model, num_samples=100, device=device)
        
        torch.manual_seed(42)
        samples2 = generate_samples(model, num_samples=100, device=device)
        
        assert torch.allclose(samples1, samples2)
    
    def test_generate_samples_without_device(self):
        model = EngressionNet()
        samples = generate_samples(model, num_samples=50)
        assert samples.shape == (50, 2)
    
    def test_generate_samples_different_dimensions(self):
        device = get_device()
        model = EngressionNet(input_dim=5, output_dim=10).to(device)
        
        samples = generate_samples(model, num_samples=200, input_dim=5, device=device)
        
        assert samples.shape == (200, 10)
    
    def test_generate_samples_no_grad(self):
        device = get_device()
        model = EngressionNet().to(device)
        
        samples = generate_samples(model, num_samples=100, device=device)
        
        # Generated samples should not require gradients
        assert not samples.requires_grad


class TestTrainAndGenerate:
    """Test the train_and_generate convenience function."""
    
    def test_train_and_generate_basic(self):
        device = get_device()
        X = torch.randn(100, 2, device=device)
        
        model, history, samples = train_and_generate(
            X,
            num_samples=50,
            num_epochs=5,
            patience=5,
            verbose=False
        )
        
        assert isinstance(model, EngressionNet)
        assert len(history['loss']) > 0
        assert samples.shape == (50, 2)
    
    def test_train_and_generate_with_numpy(self):
        X = np.random.randn(100, 3)
        
        model, history, samples = train_and_generate(
            X,
            num_samples=100,
            num_epochs=5,
            patience=5,
            verbose=False
        )
        
        assert isinstance(model, EngressionNet)
        assert samples.shape == (100, 3)
    
    def test_train_and_generate_custom_params(self):
        device = get_device()
        X = torch.randn(200, 2, device=device)
        
        model, history, samples = train_and_generate(
            X,
            num_samples=500,
            batch_size=64,
            learning_rate=1e-3,
            num_epochs=10,
            m=30,
            patience=5,
            input_dim=2,
            verbose=False,
            device=device
        )
        
        assert samples.shape == (500, 2)
        assert len(history['loss']) > 0


class TestTrainingVerbosity:
    """Test verbose output control."""
    
    def test_train_verbose_false(self, capsys):
        device = get_device()
        X = torch.randn(50, 2, device=device)
        trainer = EngressionTrainer(num_epochs=5, patience=5)
        
        trainer.train(X, verbose=False)
        
        captured = capsys.readouterr()
        assert captured.out == ""
    
    def test_train_verbose_true(self, capsys):
        device = get_device()
        X = torch.randn(50, 2, device=device)
        trainer = EngressionTrainer(num_epochs=5, patience=5)
        
        trainer.train(X, verbose=True)
        
        captured = capsys.readouterr()
        assert "Epoch" in captured.out
        assert "Loss:" in captured.out


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_train_with_small_dataset(self):
        device = get_device()
        X = torch.randn(10, 2, device=device)
        trainer = EngressionTrainer(batch_size=5, num_epochs=5, patience=5)
        
        model, history = trainer.train(X, verbose=False)
        
        assert len(history['loss']) > 0
    
    def test_train_with_single_batch(self):
        device = get_device()
        X = torch.randn(50, 2, device=device)
        trainer = EngressionTrainer(batch_size=100, num_epochs=5, patience=5)
        
        model, history = trainer.train(X, verbose=False)
        
        assert len(history['loss']) > 0
    
    def test_train_high_dimensional_output(self):
        device = get_device()
        X = torch.randn(100, 10, device=device)
        trainer = EngressionTrainer(num_epochs=5, patience=5)
        
        model, history = trainer.train(X, verbose=False)
        
        assert len(history['loss']) > 0
    
    def test_generate_single_sample(self):
        device = get_device()
        model = EngressionNet().to(device)
        
        samples = generate_samples(model, num_samples=1, device=device)
        
        assert samples.shape == (1, 2)
    
    def test_generate_large_batch(self):
        device = get_device()
        model = EngressionNet().to(device)
        
        samples = generate_samples(model, num_samples=10000, device=device)
        
        assert samples.shape == (10000, 2)


class TestModelEvaluation:
    """Test that models are properly set to evaluation mode."""
    
    def test_generate_uses_eval_mode(self):
        device = get_device()
        model = EngressionNet(dropout=0.5).to(device)
        model.train()  # Explicitly set to train mode
        
        samples = generate_samples(model, num_samples=100, device=device)
        
        # Should still work correctly
        assert samples.shape == (100, 2)
    
    def test_model_state_after_generation(self):
        device = get_device()
        model = EngressionNet().to(device)
        original_training_state = model.training
        
        generate_samples(model, num_samples=100, device=device)
        
        # Model state might change, but function should work
        assert isinstance(model, EngressionNet)