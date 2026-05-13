import matplotlib.pyplot as plt
import numpy as np

def create_ultimate_summary():
    # Setup styling - Ultra Dark "Midnight" theme
    plt.style.use('dark_background')
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 7))
    fig.patch.set_facecolor('#080808') # True black for ultimate contrast
    
    tasks = ['Game of 24', 'GSM8K', 'MATH']
    y_pos = np.arange(len(tasks))
    
    # --- Data ---
    acc_tot = [46.7, 100.0, 88.0]
    times = [51.5, 11.9, 297.0]
    calls = [68.5, 6.0, 154.0]
    
    # --- 1. Accuracy Panel (Deep Emerald) ---
    ax1.set_facecolor('#080808')
    bars1 = ax1.barh(y_pos, acc_tot, color='#0a4d4a', edgecolor='#1abc9c', linewidth=1.5, zorder=3)
    ax1.set_title('Final Accuracy (%)', fontsize=16, fontweight='bold', color='#1abc9c', pad=20)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(tasks, fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 115)
    ax1.xaxis.grid(True, color='#222', linestyle='--', alpha=0.5)
    
    for bar in bars1:
        w = bar.get_width()
        ax1.text(w + 3, bar.get_y() + bar.get_height()/2, f'{w:.1f}%', 
                va='center', color='#1abc9c', fontweight='bold', fontsize=12)

    # --- 2. Latency Panel (Burnt Amber) ---
    ax2.set_facecolor('#080808')
    bars2 = ax2.barh(y_pos, times, color='#5a3a0a', edgecolor='#e67e22', linewidth=1.5, zorder=3)
    ax2.set_title('Latency (Seconds)', fontsize=16, fontweight='bold', color='#e67e22', pad=20)
    ax2.set_yticks([]) # Hide Y labels for middle panel
    ax2.xaxis.grid(True, color='#222', linestyle='--', alpha=0.5)
    
    for bar in bars2:
        w = bar.get_width()
        # Handle large numbers gracefully
        label = f'{w:.1f}s' if w < 100 else f'{w:.0f}s'
        ax2.text(w + 5, bar.get_y() + bar.get_height()/2, label, 
                va='center', color='#e67e22', fontweight='bold', fontsize=12)

    # --- 3. API Calls Panel (Deep Purple) ---
    ax3.set_facecolor('#080808')
    bars3 = ax3.barh(y_pos, calls, color='#3a0a5a', edgecolor='#9b59b6', linewidth=1.5, zorder=3)
    ax3.set_title('API Call Volume', fontsize=16, fontweight='bold', color='#9b59b6', pad=20)
    ax3.set_yticks([]) # Hide Y labels for right panel
    ax3.xaxis.grid(True, color='#222', linestyle='--', alpha=0.5)
    
    for bar in bars3:
        w = bar.get_width()
        ax3.text(w + 3, bar.get_y() + bar.get_height()/2, f'{w:.0f}', 
                va='center', color='#9b59b6', fontweight='bold', fontsize=12)

    # Global Styling
    for ax in [ax1, ax2, ax3]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#333')

    plt.suptitle('THE TREE OF THOUGHTS MASTER BENCHMARK', 
                 fontsize=24, fontweight='bold', color='white', y=1.05)
    plt.tight_layout()
    
    plt.savefig('ultimate_master_plot.png', dpi=300, bbox_inches='tight', transparent=True)
    print("Ultimate master plot saved as ultimate_master_plot.png")

if __name__ == '__main__':
    create_ultimate_summary()
