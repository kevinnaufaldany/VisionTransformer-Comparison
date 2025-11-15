# train_deit.py
import torch
from datareader import get_food_dataset
from model import get_model, count_parameters
from utils import train_model, evaluate_metrics, measure_inference_time, check_set_gpu, ResultsLogger

if __name__ == "__main__":
    device = check_set_gpu()

    # Load custom food dataset
    trainloader, valloader, label2idx, num_classes = get_food_dataset(
        batch_size=32,
        img_size=224,
        data_dir='dataset',
        csv_file='dataset.csv'
    )

    # Create reverse mapping (idx to label)
    idx2label = {v: k for k, v in label2idx.items()}
    class_names = [idx2label[i] for i in range(num_classes)]

    model = get_model("deit", num_classes=num_classes)

    total, trainable, nontrainable = count_parameters(model)
    
    # Initialize results logger
    logger = ResultsLogger(model_name="DeiT", num_classes=num_classes)
    logger.log_parameters(model, trainable, total, nontrainable)
    logger.log_hardware()

    # Train model dengan logger untuk save best weights
    train_losses, val_losses, val_accs, model = train_model(
        model, trainloader, valloader, device,
        model_name="DeiT",
        epochs=10,
        logger=logger
    )
    logger.log_training_metrics(train_losses, val_losses, val_accs)

    # Evaluate metrics
    precision, recall, f1, trues, preds = evaluate_metrics(model, valloader, device)
    logger.log_performance_metrics(trues, preds, class_names=class_names)

    # Measure inference time
    avg_ms, throughput = measure_inference_time(model, valloader, device)
    device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    logger.log_inference_metrics(avg_ms, throughput, device_name)

    # Save all results
    logger.save_history()
    logger.print_summary()
