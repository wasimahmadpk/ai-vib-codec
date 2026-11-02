def save_codec(
    path: Path,
    model: Autoencoder,
    mean: float,
    std: float,
    rpm_mean: float,
    rpm_std: float,
    side_info: bool = False,
    window: int = 1,
    compress_code: bool = False,
) -> None:
    """Store the shared weights and the train-set normalization."""
    # Ensure the target directory exists before writing the serialized model.
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            # Saved model parameters needed to rebuild the same encoder/decoder.
            "state": model.state_dict(),
            "code_channels": model.code_channels,
            "code_time": model.code_time,
            # Normalization values used during training/encoding/decoding.
            "mean": float(mean),
            "std": float(std),
            "rpm_mean": float(rpm_mean),
            "rpm_std": float(rpm_std),
            # Codec settings that affect packet format and reconstruction behavior.
            "side_info": bool(side_info),
            "window": int(window),
            "compress_code": bool(compress_code),
        },
        path,
    )


def open_codec(path: Path, device: str | torch.device = "cpu") -> Codec:
    """Load a trained codec onto one device."""
    # Load the saved checkpoint and map tensors to the requested device.
    stored = torch.load(path, map_location=device, weights_only=False)
    # Recreate the architecture with the same code dimensionality as the saved model.
    model = Autoencoder(
        int(stored["code_channels"]),
        int(stored.get("code_time", CODE_TIME)),
    )
    # Restore the trained weights into the new model instance.
    model.load_state_dict(stored["state"])
    model.to(device)
    model.eval()
    # Build the runtime Codec wrapper with saved normalization and packet settings.
    return Codec(
        model,
        stored["mean"],
        stored["std"],
        stored["rpm_mean"],
        stored["rpm_std"],
        side_info=bool(stored.get("side_info", False)),
        window=int(stored.get("window", 1)),
        compress_code=bool(stored.get("compress_code", False)),
    )
