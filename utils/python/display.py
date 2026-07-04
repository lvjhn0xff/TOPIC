
def display_confusion_matrix(cm, labels, prefix=">>> "):
    """
    Prints a labeled confusion matrix where every single line 
    starts with a custom string prefix.
    """
    # Find the maximum width of labels and numbers for clean alignment
    max_label_len = max(len(str(label)) for label in labels)
    col_width = max(max_label_len, 5) 

    # 1. Format the header row
    header_gap = " " * (max_label_len + 3)
    header_cols = "".join([f"{label:>{col_width}}" for label in labels])
    print(f"{prefix}{header_gap}{header_cols}")
    
    # 2. Format the divider line
    total_width = len(header_gap) + len(header_cols)
    print(f"{prefix}{'-' * total_width}")

    # 3. Format the data rows
    for i, true_label in enumerate(labels):
        row_str = f"{prefix}{true_label:<{max_label_len}} | "
        for j in range(len(labels)):
            row_str += f"{cm[i, j]:>{col_width}}"
        print(row_str)