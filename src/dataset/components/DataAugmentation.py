from torchvision import transforms

def ImageAugmentation():
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),  # Đảo ngược ảnh theo chiều ngang
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Tăng cường màu sắc
            transforms.RandomRotation(degrees=15),  # Xoay ảnh một góc ngẫu nhiên
            transforms.Resize(224),  # resize ảnh
            transforms.ToTensor(),  # Chuyển ảnh thành tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Chuẩn hóa dữ liệu
        ]
    )