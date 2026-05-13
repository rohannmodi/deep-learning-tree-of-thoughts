import matplotlib.pyplot as plt
import numpy as np

def create_level_plot():
    # Setup styling
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('#1e1e2e')
    ax.set_facecolor('#1e1e2e')

    # Data
    levels = ['Level 1', 'Level 2', 'Level 3', 'Level 4', 'Level 5']
    ss_acc = [100.0, 100.0, 80.0, 60.0, 80.0]
    cot_acc = [100.0, 100.0, 80.0, 80.0, 80.0]
    tot_acc = [100.0, 100.0, 80.0, 80.0, 80.0]
    
    x = np.arange(len(levels))
    width = 0.25

    # Colors
    color_ss = '#4facfe'
    color_cot = '#00f2fe'
    color_tot = '#00c6ff'

    # Plot grouped bars
    bars_ss = ax.bar(x - width, ss_acc, width, label='Single-Shot', color=color_ss, zorder=3)
    bars_cot = ax.bar(x, cot_acc, width, label='Chain of Thought', color=color_cot, zorder=3)
    bars_tot = ax.bar(x + width, tot_acc, width, label='Tree of Thoughts', color=color_tot, zorder=3)

    # Styling axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cdd6f4')
    ax.spines['bottom'].set_color('#cdd6f4')
    ax.tick_params(colors='#cdd6f4', labelsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.set_ylim(0, 120)

    # Add horizontal gridlines behind bars
    ax.yaxis.grid(True, color='#313244', linestyle='dashed', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Add text labels on top of bars
    for bars in [bars_ss, bars_cot, bars_tot]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.0f}%',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5),  # 5 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold', color='white', rotation=90)

    # Add title and labels
    ax.set_title('Gemini 2.5 Flash: MATH Accuracy by Prompting Strategy', 
                 fontsize=18, fontweight='bold', color='white', pad=20)
    ax.set_ylabel('Accuracy (%)', fontsize=14, color='#cdd6f4', labelpad=10)
    
    # Add a legend
    ax.legend(loc='upper right', frameon=False, labelcolor='white', fontsize=12)

    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_filename = 'math_levels_plot.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Plot saved to: {output_filename}")

if __name__ == '__main__':
    create_level_plot()
