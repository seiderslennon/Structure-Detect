from ai_music.infer import load_model_from_checkpoint, create_inference_dataloader, trainer_inference

# Load model
model = load_model_from_checkpoint("best_model.ckpt")

# Create dataloader
dataloader = create_inference_dataloader(csv_path="test_data.csv")

# Run inference
predictions = trainer_inference(model, dataloader)

# Process results
for batch_pred in predictions:
    for pred, prob in zip(batch_pred["predictions"], batch_pred["probabilities"]):
        confidence = max(prob)
        print(f"Prediction: {pred}, Confidence: {confidence:.4f}")


# # Basic inference with Lightning Trainer (recommended)
# python infer.py --checkpoint best_model.ckpt --csv test_data.csv --use_trainer

# # Show detailed probabilities
# python infer.py --checkpoint best_model.ckpt --csv test_data.csv --use_trainer --detailed

# # Save results to CSV
# python infer.py --checkpoint best_model.ckpt --csv test_data.csv --use_trainer --output results.csv