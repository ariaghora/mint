from torchvision import models
import torch

if __name__ == "__main__":
    model_path_in = "resnet-18.onnx"

    # Load the pre-trained AlexNet model
    resnet = models.resnet18(pretrained=True)

    # Set the model to evaluation mode
    resnet.eval()

    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224)

    # Export the model to ONNX
    torch.onnx.export(
        resnet,  
        dummy_input,  
        model_path_in,  
        export_params=True, 
        opset_version=12,  
        do_constant_folding=True, 
        input_names=["input"],  
        output_names=["output"], 
        dynamic_axes={
            "input": {0: "batch_size"},  
            "output": {0: "batch_size"},
        },
    )
