import json
import re

nb_paths = [
    "/Users/gimdabin/SKN23-2nd-3Team/notebooks/dl/MLP_enhance.ipynb",
    "/Users/gimdabin/SKN23-2nd-3Team/notebooks/dl/MLP_advanced.ipynb",
]


def revert_filenames():
    for nb_path in nb_paths:
        with open(nb_path, "r", encoding="utf-8") as f:
            nb = json.load(f)

        cells = nb["cells"]

        # Determine model name base string based on filename
        model_name_base = "mlp_enhance" if "enhance" in nb_path else "mlp_advanced"

        for cell in cells:
            if cell["cell_type"] == "code" and "torch.save(" in "".join(cell["source"]):
                source = list(cell["source"])
                new_source = []
                for line in source:
                    # Check for our dynamic filename pattern
                    # e.g. f"mlp_enhance_opt_{optimizer_name}_act_{activation}.pt"

                    if "torch.save(" in line and ".pt" in line:
                        # Revert to fixed name
                        # pattern: anything inside os.path.join(MODEL_DIR, ...)
                        # We'll just force the line to use the fixed name
                        line = f'torch.save(model.state_dict(), os.path.join(MODEL_DIR, "{model_name_base}.pt"))\\n'
                        new_source.append(line)

                    elif "Saved Model:" in line and ".pt" in line:
                        line = f"print(f\"Saved Model: {{os.path.join(MODEL_DIR, '{model_name_base}.pt')}}\")\\n"
                        new_source.append(line)

                    # Also check for metrics/scaler if I modified them (I might not have, but let's be safe)
                    # I didn't modify scaler/metrics filenames in previous steps, but user asked to overwrite existing ones.
                    # The "Smart Saving" I implemented only touched the .pt file name mainly.
                    # But I should double check if I touched others.
                    # In inline_tuning.py I didn't touch others.
                    # In fix_notebook_final.py I looked for "mlp_enhance.pt" pattern.

                    # Wait, the user said "gijon basic model and json files".
                    # Means overwrite existing. The existing logic ALREADY overwrites if using fixed names.
                    # My previous change made them dynamic. Now returning to fixed.

                    else:
                        new_source.append(line)

                cell["source"] = new_source

        with open(nb_path, "w", encoding="utf-8") as f:
            json.dump(nb, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    revert_filenames()
