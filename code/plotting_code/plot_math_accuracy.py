import matplotlib.pyplot as plt
import numpy as np

def create_math_plot():
    # Setup styling
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1e1e2e')
    ax.set_facecolor('#1e1e2e')

    # Data
    methods = ['Single-Shot (IO)', 'Chain of Thought', 'Tree of Thoughts']
    accuracy = [84.0, 88.0, 88.0]
    
    # Modern color palette
    colors = ['#b8c0ff', '#c8b6ff', '#e7c6ff']

    # Plot bars
    bars = ax.bar(methods, accuracy, color=colors, width=0.6, zorder=3)

    # Styling axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#cdd6f4')
    ax.spines['bottom'].set_color('#cdd6f4')
    ax.tick_params(colors='#cdd6f4', labelsize=12)
    ax.set_ylim(0, 110)

    # Add horizontal gridlines behind bars
    ax.yaxis.grid(True, color='#313244', linestyle='dashed', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)

    # Add text labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),  # 5 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=14, fontweight='bold', color='white')

    # Add title and labels
    ax.set_title('Gemini 2.5 Flash: MATH Accuracy (N=25)', 
                 fontsize=18, fontweight='bold', color='white', pad=20)
    ax.set_ylabel('Accuracy (%)', fontsize=14, color='#cdd6f4', labelpad=10)
    
    # Add an annotation explaining the ceiling effect
    ax.text(1, 100, "Knowledge Ceiling:\nToT matches CoT/IO when bounded\nby intrinsic mathematical knowledge.",
            ha='center', va='top', fontsize=12, color='#f38ba8',
            bbox=dict(facecolor='#1e1e2e', edgecolor='#f38ba8', boxstyle='round,pad=0.5', alpha=0.9))

    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_filename = 'math_accuracy_plot.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Plot saved to: {output_filename}")

if __name__ == '__main__':
    create_math_plot()
