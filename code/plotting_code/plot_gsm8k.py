import matplotlib.pyplot as plt
import numpy as np

def create_gsm8k_plot():
    # Setup styling
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1e1e2e')
    ax.set_facecolor('#1e1e2e')

    # Data
    methods = ['Single-Shot (IO)', 'Chain of Thought', 'Tree of Thoughts']
    accuracy = [85.0, 90.0, 100.0]
    
    # Modern color palette (vibrant gradient-like colors)
    colors = ['#ff7eb3', '#ff758c', '#ff7eb3'] # pinkish
    colors = ['#4facfe', '#00f2fe', '#4facfe'] # blueish
    colors = ['#fa709a', '#fee140', '#00c6ff'] # dynamic different colors
    colors = ['#b8c0ff', '#c8b6ff', '#e7c6ff'] # soft modern
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1'] # punchy

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
    ax.set_title('Gemini 2.5 Flash: GSM8K Accuracy Scaling', 
                 fontsize=18, fontweight='bold', color='white', pad=20)
    ax.set_ylabel('Accuracy (%)', fontsize=14, color='#cdd6f4', labelpad=10)

    # Adjust layout
    plt.tight_layout()
    
    # Save the figure
    output_filename = 'gsm8k_accuracy_plot.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    print(f"Plot saved to: {output_filename}")

if __name__ == '__main__':
    create_gsm8k_plot()
