import subprocess
from itertools import product

# Define hyperparameter search space
learning_rates = [0.001, 0.005, 0.01]
hidden_dims = [8, 16, 32]
dropout_rates = [0.0, 0.1]

# Variables to track the best hyperparameters and results
best_val_acc = 0.0
best_hyperparams = None
best_test_acc = 0.0

for lr, hidden_dim, dropout in product(learning_rates, hidden_dims, dropout_rates):
    command = [
        "/opt/homebrew/bin/python3.10",
        "train.py",
        "--bmname=synthetic",
        f"--lr={lr}",
        f"--hidden-dim={hidden_dim}",
        f"--dropout={dropout}",
        f"--epochs={50}",  # Adjust the number of epochs as needed
        f"--batch_size={64}",
    ]

    print(f"Running train.py with lr={lr}, hidden_dim={hidden_dim}, dropout={dropout}")

    # Run the training script
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output = result.stdout.decode("utf-8")

    # Print the result for debugging/logging purposes
    print(output)

    if result.returncode != 0:
        print(f"Error occurred: {result.stderr.decode('utf-8')}")
        continue

    # Extract the last occurrence of validation and test accuracy from the output
    try:
        # Find all occurrences of Validation and Test accuracy lines
        val_acc_lines = [
            line for line in output.splitlines() if "Validation  accuracy:" in line
        ]
        test_acc_lines = [
            line for line in output.splitlines() if "Test  accuracy:" in line
        ]

        # Get the last occurrence of the accuracy lines
        if val_acc_lines and test_acc_lines:
            last_val_acc_line = val_acc_lines[-1]
            last_test_acc_line = test_acc_lines[-1]

            # Extract the accuracy values
            val_acc = float(last_val_acc_line.split("Validation  accuracy: ")[1])
            test_acc = float(last_test_acc_line.split("Test  accuracy: ")[1])

            print(
                f"Final Validation accuracy: {val_acc}, Final Test accuracy: {test_acc}"
            )

            # Check if the current hyperparameters perform better on the validation set
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_test_acc = test_acc
                best_hyperparams = {
                    "lr": lr,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                }

    except (IndexError, ValueError) as e:
        print(f"Error parsing accuracies: {e}")

# Print the best hyperparameters and corresponding accuracies
if best_hyperparams:
    print("\nBest Hyperparameters:")
    print(f"Learning Rate: {best_hyperparams['lr']}")
    print(f"Hidden Dimension: {best_hyperparams['hidden_dim']}")
    print(f"Dropout: {best_hyperparams['dropout']}")
    print(f"Best Validation Accuracy: {best_val_acc}")
    print(f"Best Test Accuracy: {best_test_acc}")
else:
    print("No valid results were found.")
