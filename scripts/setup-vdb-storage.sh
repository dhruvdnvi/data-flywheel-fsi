#!/bin/bash
set -e

# Display usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Setup Docker and Minikube storage on a specified device and mount point.

OPTIONS:
    -m, --mount-point PATH    Mount point path (e.g., /mnt/sda2)
    -d, --device PATH         Device path (e.g., /dev/sda2)
    -h, --help                Display this help message

EXAMPLES:
    # Interactive mode (prompts for input)
    $0

    # Specify mount point (device auto-derived)
    $0 -m /mnt/nvme0n1

    # Specify both mount point and device
    $0 -m /mnt/data -d /dev/nvme0n1

    # Quick positional argument
    $0 /mnt/sda2

DESCRIPTION:
    This script configures Docker and Minikube to use a specified storage device.
    It will format the device, mount it, and configure both Docker and Minikube
    to store their data on the mounted volume with NVIDIA runtime support.

EOF
    exit 0
}

# Default values
MOUNT_POINT=""
DEVICE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--mount-point)
            MOUNT_POINT="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        -*)
            echo "Error: Unknown option $1"
            usage
            ;;
        *)
            # Treat first positional argument as mount point
            if [ -z "$MOUNT_POINT" ]; then
                MOUNT_POINT="$1"
            fi
            shift
            ;;
    esac
done

# Prompt for mount point if not provided
if [ -z "$MOUNT_POINT" ]; then
    read -p "Enter mount point (default: /mnt/sda2): " MOUNT_POINT
    MOUNT_POINT="${MOUNT_POINT:-/mnt/sda2}"
fi

# Derive device name from mount point if not provided
if [ -z "$DEVICE" ]; then
    DEVICE="/dev/$(basename "$MOUNT_POINT")"
    echo "Auto-derived device from mount point: $DEVICE"
fi

echo "================================================"
echo "Configuration:"
echo "  Device:      $DEVICE       (physical storage hardware)"
echo "  Mount Point: $MOUNT_POINT (directory to access the device)"
echo "================================================"
echo ""
echo "What this means:"
echo "  - The device $DEVICE will be formatted with ext4 filesystem"
echo "  - It will be mounted (made accessible) at $MOUNT_POINT"
echo "  - Docker and Minikube data will be stored on this device"
echo ""

# Confirm before proceeding
read -p "Proceed with setup? (y/n): " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Setup cancelled."
    exit 1
fi

# Mount the device if not already mounted
if ! mountpoint -q "$MOUNT_POINT"; then
    echo "Setting up storage..."
    # Create mount point directory (safe even if exists due to -p flag)
    sudo mkdir -p "$MOUNT_POINT"
    # Format the device with ext4 filesystem
    sudo mkfs.ext4 -F "$DEVICE" 2>/dev/null || true
    # Mount the device to the mount point
    sudo mount "$DEVICE" "$MOUNT_POINT"
    # Add to fstab for automatic mounting on boot
    echo "$DEVICE $MOUNT_POINT ext4 defaults 0 2" | sudo tee -a /etc/fstab
    echo "✓ Device $DEVICE mounted at $MOUNT_POINT"
else
    echo "✓ $MOUNT_POINT is already mounted"
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
