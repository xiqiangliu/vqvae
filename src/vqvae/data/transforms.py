import torchvision.transforms as tv_transforms

standard_transform = tv_transforms.Compose(
    [
        tv_transforms.ToTensor(),
        tv_transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)
