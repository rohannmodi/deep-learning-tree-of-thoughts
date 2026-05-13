import matplotlib.pyplot as plt
import os

def create_accuracy_table():
    columns = ('Task', 'Method', 'Original Paper (ToT)', 'Gemini 2.5 Flash')
    cell_text = [
        ['Crossword', 'Letter-level (ToT)', '78.0%', '69.6%'],
        ['Crossword', 'Word-level (ToT)', '60.0%', '44.0%'],
        ['Game of 24', 'Tree of Thoughts (ToT)', '74.0%', '46.7%'],
        ['GSM8K', 'Input-Output (IO)', '-', '85.0%'],
        ['GSM8K', 'Chain of Thought (CoT)', '-', '90.0%'],
        ['GSM8K', 'Tree of Thoughts (ToT)', '-', '100.0%'],
        ['MATH', 'Input-Output (IO)', '-', '84.0%'],
        ['MATH', 'Chain of Thought (CoT)', '-', '88.0%'],
        ['MATH', 'Tree of Thoughts (ToT)', '-', '88.0%']
    ]

    fig, ax = plt.subplots(figsize=(12, 6.0))
    ax.axis('tight')
    ax.axis('off')

    # Create table
    table = ax.table(cellText=cell_text, colLabels=columns, loc='center', cellLoc='center')
    
    # Auto-adjust column widths to prevent text clipping
    table.auto_set_column_width(col=list(range(len(columns))))

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.5)

    # Header formatting
    for j in range(len(columns)):
        cell = table[0, j]
        cell.set_text_props(weight='bold', color='white')
        cell.set_facecolor('#2c3e50')

    # Alternating row colors
    for i in range(1, len(cell_text) + 1):
        for j in range(len(columns)):
            cell = table[i, j]
            if i % 2 == 0:
                cell.set_facecolor('#f8f9fa')
            else:
                cell.set_facecolor('#ffffff')

    plt.title("Tree of Thoughts Benchmarks: Paper vs. Gemini 2.5 Flash", fontsize=16, fontweight='bold', pad=20)
    
    out_path = os.path.join(os.path.dirname(__file__), "accuracy_table.png")
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Table saved to: {out_path}")

if __name__ == '__main__':
    create_accuracy_table()
