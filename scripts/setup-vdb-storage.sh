#!/bin/bash
set -e

MOUNT_POINT="/mnt/vdb"
DEVICE="/dev/sda2"

# Mount /dev/vdb if not already mounted
if ! mountpoint -q "$MOUNT_POINT"; then
    sudo mkdir -p "$MOUNT_POINT"
    sudo mkfs.ext4 -F "$DEVICE" 2>/dev/null || true
    sudo mount "$DEVICE" "$MOUNT_POINT"
    echo "$DEVICE $MOUNT_POINT ext4 defaults 0 2" | sudo tee -a /etc/fstab
fi

# Configure Docker with NVIDIA runtime
sudo systemctl stop docker 2>/dev/null || true
sudo mkdir -p "$MOUNT_POINT/docker"
[ -d "/var/lib/docker" ] && sudo rsync -aP /var/lib/docker/ "$MOUNT_POINT/docker/" 2>/dev/null || true

sudo tee /etc/docker/daemon.json > /dev/null <<EOF
{
  "data-root": "$MOUNT_POINT/docker",
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia"
}
EOF

sudo systemctl start docker

# Configure Minikube
minikube stop 2>/dev/null || true
sudo mkdir -p "$MOUNT_POINT/minikube"
sudo chown -R $USER:$USER "$MOUNT_POINT/minikube"
[ -d "$HOME/.minikube" ] && [ ! -L "$HOME/.minikube" ] && rsync -aP "$HOME/.minikube/" "$MOUNT_POINT/minikube/" 2>/dev/null || true
ln -sf "$MOUNT_POINT/minikube" "$HOME/.minikube"

echo "✓ Setup complete: Docker & Minikube using /dev/vdb with NVIDIA runtime"
df -h "$MOUNT_POINT"
