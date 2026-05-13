import matplotlib.pyplot as plt
import numpy as np

def create_unfolded_token_plot():
    plt.style.use('default')
    
    # --- DATA ---
    labels = ['G24', 'GSM8K', 'M-L1', 'M-L2', 'M-L3', 'M-L4', 'M-L5']
    
    # Token Counts
    # G24: 23,975
    # GSM8K: 2,400
    # MATH Levels (Calls * 500 tokens)
    # L1: 1.0 * 500 = 500
    # L2: 1.0 * 500 = 500
    # L3: 10.5 * 500 = 5250
    # L4: 45.2 * 500 = 22600
    # L5: 35.8 * 500 = 17900
    tokens = [23975, 2400, 500, 500, 5250, 22600, 17900]
    
    # Purple gradient theme
    colors = ['#6a1b9a', '#8e24aa', '#e1bee7', '#ce93d8', '#ba68c8', '#ab47bc', '#9c27b0']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    bars = ax.bar(labels, tokens, color=colors, edgecolor='black', width=0.7, zorder=3)
    
    ax.set_title('Token Throughput: Dataset Comparison (Unfolded)', fontsize=18, fontweight='bold', pad=20)
    ax.set_ylabel('Total Tokens per Task', fontsize=14, fontweight='bold')
    
    # Styling
    ax.yaxis.grid(True, color='#eee', linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(labelsize=12)
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    
    # Value labels
    for bar in bars:
        h = bar.get_height()
        label = f'{h/1000:.1f}k' if h >= 1000 else f'{h:.0f}'
        ax.text(bar.get_x() + bar.get_width()/2, h + 500, label, ha='center', color='black', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plt.savefig('plot_tokens_unfolded.png', dpi=300, bbox_inches='tight')
    print("Unfolded token plot saved.")

if __name__ == '__main__':
    create_unfolded_token_plot()
