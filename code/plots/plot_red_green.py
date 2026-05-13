import matplotlib.pyplot as plt
import numpy as np

def create_red_green_plots():
    plt.style.use('dark_background')
    tasks = ['Game of 24', 'GSM8K', 'MATH']
    
    # --- DATA ---
    acc_tot = [46.7, 100.0, 88.0]
    times = [51.5, 11.9, 297.0]
    calls = [68.5, 6.0, 154.0]
    
    # ---------------------------------------------------------
    # PLOT 1: Accuracy (Deep Green - Success)
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    fig1.patch.set_facecolor('#080808')
    ax1.set_facecolor('#080808')
    
    bars1 = ax1.bar(tasks, acc_tot, color='#144026', edgecolor='#2ecc71', width=0.6, zorder=3)
    
    ax1.set_title('ToT Success Rates', fontsize=16, fontweight='bold', color='#2ecc71', pad=15)
    ax1.set_ylabel('Accuracy (%)', color='#cdd6f4', fontsize=12)
    ax1.set_ylim(0, 115)
    ax1.yaxis.grid(True, color='#222', linestyle='--', alpha=0.6)
    
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 2, f'{h:.1f}%', 
                ha='center', color='#2ecc71', fontweight='bold', fontsize=11)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.savefig('accuracy_green.png', dpi=300, bbox_inches='tight', transparent=True)

    # ---------------------------------------------------------
    # PLOT 2: Resource Cost (Deep Red - Expense)
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    fig2.patch.set_facecolor('#080808')
    ax2.set_facecolor('#080808')
    
    x = np.arange(len(tasks))
    width = 0.35
    
    # Using two shades of Red for Time and Calls
    b_times = ax2.bar(x - width/2, times, width, label='Time (s)', color='#4a0e0e', edgecolor='#e74c3c', zorder=3)
    b_calls = ax2.bar(x + width/2, calls, width, label='API Calls', color='#6e1212', edgecolor='#ff4d4d', zorder=3)
    
    ax2.set_title('Computational Expense', fontsize=16, fontweight='bold', color='#e74c3c', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks, fontsize=12, fontweight='bold')
    ax2.set_ylabel('Resource Units', color='#cdd6f4', fontsize=12)
    ax2.yaxis.grid(True, color='#222', linestyle='--', alpha=0.6)
    ax2.legend(frameon=False, labelcolor='white', fontsize=11)
    
    for bar in b_times:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 5, f'{h:.0f}s', ha='center', color='#e74c3c', fontweight='bold')
    for bar in b_calls:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 5, f'{h:.0f}', ha='center', color='#ff4d4d', fontweight='bold')

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    plt.savefig('cost_red.png', dpi=300, bbox_inches='tight', transparent=True)
    
    print("Green and Red plots saved.")

if __name__ == '__main__':
    create_red_green_plots()
