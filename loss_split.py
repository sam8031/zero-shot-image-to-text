import re

# Extracts losses from an input file and writes them to an output file
def extract_losses(input_file, output_file):
    # Store extracted losses
    losses = []

    with open(input_file, 'r', encoding='utf-8') as infile:
        for line in infile:
            # Use regular expression to search for loss pattern in the line
            match = re.search(r'Average Loss for Epoch \d+: ([0-9]+\.[0-9]{5,})', line)
            if match:
                # If a match is found, extract the loss value and append to the losses list
                loss = match.group(1)
                losses.append(loss)

    # Output file for writing
    with open(output_file, 'w', encoding='utf-8') as outfile:
        # Write the extracted losses to the output file, separated by commas
        outfile.write(',\n'.join(losses))

input_file_path = r"./zero-shot-image-to-text/output1.txt"
output_file_path = r"C:\Users\danie\PycharmProjects\script line losses\losses.txt"

extract_losses(input_file_path, output_file_path)
