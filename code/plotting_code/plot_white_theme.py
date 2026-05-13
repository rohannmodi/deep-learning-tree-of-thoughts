import matplotlib.pyplot as plt
import numpy as np

def create_black_label_plots():
    # Use default light style
    plt.style.use('default')
    tasks = ['Game of 24', 'GSM8K', 'MATH']
    
    # --- DATA ---
    acc_tot = [46.7, 100.0, 88.0]
    times = [51.5, 11.9, 297.0]
    calls = [68.5, 6.0, 154.0]
    
    # ---------------------------------------------------------
    # PLOT 1: Accuracy (Green) - White Theme
    # ---------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    fig1.patch.set_facecolor('white')
    ax1.set_facecolor('white')
    
    bars1 = ax1.bar(tasks, acc_tot, color='#27ae60', edgecolor='black', width=0.6, zorder=3)
    
    ax1.set_title('ToT Success Rates', fontsize=16, fontweight='bold', color='black', pad=15)
    ax1.set_ylabel('Accuracy (%)', color='black', fontsize=12, fontweight='bold')
    ax1.set_ylim(0, 115)
    ax1.yaxis.grid(True, color='#eee', linestyle='--', zorder=0)
    
    # X/Y labels and ticks to black
    ax1.tick_params(colors='black', labelsize=12)
    for label in ax1.get_xticklabels():
        label.set_fontweight('bold')
        label.set_color('black')
        
    for bar in bars1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 2, f'{h:.1f}%', 
                ha='center', color='black', fontweight='bold', fontsize=11)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('black')
    ax1.spines['bottom'].set_color('black')
    
    plt.savefig('accuracy_green_white.png', dpi=300, bbox_inches='tight')

    # ---------------------------------------------------------
    # PLOT 2: Resource Cost (Red) - White Theme
    # ---------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    fig2.patch.set_facecolor('white')
    ax2.set_facecolor('white')
    
    x = np.arange(len(tasks))
    width = 0.35
    
    b_times = ax2.bar(x - width/2, times, width, label='Time (s)', color='#c0392b', edgecolor='black', zorder=3)
    b_calls = ax2.bar(x + width/2, calls, width, label='API Calls', color='#e74c3c', edgecolor='black', zorder=3)
    
    ax2.set_title('Computational Expense (Avg)', fontsize=16, fontweight='bold', color='black', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tasks, fontsize=12, fontweight='bold', color='black')
    ax2.set_ylabel('Resource Units', color='black', fontsize=12, fontweight='bold')
    ax2.yaxis.grid(True, color='#eee', linestyle='--', zorder=0)
    
    # Legend in black
    leg = ax2.legend(frameon=False, fontsize=11)
    for text in leg.get_texts():
        text.set_color('black')
        text.set_fontweight('bold')
    
    ax2.tick_params(colors='black', labelsize=11)
    
    for bar in b_times:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 5, f'{h:.0f}s', ha='center', color='black', fontweight='bold')
    for bar in b_calls:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, h + 5, f'{h:.0f}', ha='center', color='black', fontweight='bold')

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('black')
    ax2.spines['bottom'].set_color('black')
    
    plt.savefig('cost_red_white.png', dpi=300, bbox_inches='tight')
    
    print("White theme plots saved.")

if __name__ == '__main__':
    create_black_label_plots()
