import matplotlib.pyplot as plt
import numpy as np

def create_compact_nested_plot():
    plt.style.use('default')
    
    # --- DATA ---
    # G24, GSM8K
    base_labels = ['Game of 24', 'GSM8K']
    base_tokens = [23975, 2400]
    
    # MATH Levels
    math_tokens = [500, 500, 5250, 22600, 17900]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # Main Width for standard bars
    main_width = 0.7
    
    # 1. Plot standard bars (x=0 and x=1)
    ax.bar(0, base_tokens[0], width=main_width, color='#6a1b9a', edgecolor='black', zorder=3)
    ax.bar(1, base_tokens[1], width=main_width, color='#8e24aa', edgecolor='black', zorder=3)
    
    # 2. Plot nested MATH bars (at x=2)
    # Total space for MATH = main_width
    # Width per sub-bar = main_width / 5
    sub_width = (main_width - 0.05) / 5 # Slight gap between them
    math_x_start = 2 - (main_width/2) + (sub_width/2)
    math_offsets = [math_x_start + i*(sub_width + 0.01) for i in range(5)]
    
    math_colors = ['#e1bee7', '#ce93d8', '#ba68c8', '#ab47bc', '#9c27b0']
    
    for i in range(5):
        ax.bar(math_offsets[i], math_tokens[i], width=sub_width, color=math_colors[i], edgecolor='black', zorder=3)
    
    # Labels
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['Game of 24', 'GSM8K', 'MATH (L1-L5)'], fontweight='bold', fontsize=12, color='black')
    ax.set_ylabel('Total Tokens per Task', fontsize=12, fontweight='bold', color='black')
    ax.set_title('Token Throughput: Compact Complexity Mapping', fontsize=16, fontweight='bold', color='black', pad=20)
    
    # Value Labels for standard bars
    ax.text(0, base_tokens[0] + 1000, f'{base_tokens[0]/1000:.1f}k', ha='center', fontweight='bold', color='black')
    ax.text(1, base_tokens[1] + 1000, f'{base_tokens[1]/1000:.1f}k', ha='center', fontweight='bold', color='black')
    
    # Special label for Level 4/5 peak
    ax.text(2, math_tokens[3] + 1000, f'Peak: {math_tokens[3]/1000:.1f}k', ha='center', fontweight='bold', color='black', fontsize=10)

    ax.yaxis.grid(True, color='#eee', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')
    ax.tick_params(colors='black')
    
    plt.tight_layout()
    plt.savefig('plot_tokens_compact.png', dpi=300, bbox_inches='tight')
    print("Compact nested plot saved.")

if __name__ == '__main__':
    create_compact_nested_plot()
