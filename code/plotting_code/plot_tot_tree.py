import matplotlib.pyplot as plt
import matplotlib.patches as patches
import networkx as nx

def create_tree_plot():
    # Setup styling
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(9, 12))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Create directed graph
    G = nx.DiGraph()

    # Define nodes with positions and labels
    # Format: node_id: (x, y, label, color)
    nodes = {
        'Root': (5, 4, 'Input:\n4 5 6 7', '#cba6f7'),  # Purple
        
        # Depth 1
        'N1_1': (2, 3, '5 + 7 = 12\nState: (4, 6, 12)\nEval: sure', '#a6e3a1'), # Green
        'N1_2': (5, 3, '4 * 5 = 20\nState: (6, 7, 20)\nEval: maybe', '#f9e2af'), # Yellow
        'N1_3': (8, 3, '5 - 4 = 1\nState: (1, 6, 7)\nEval: impossible', '#f38ba8'), # Red
        
        # Depth 2 (From N1_1)
        'N2_1': (1, 2, '6 - 4 = 2\nState: (2, 12)\nEval: sure', '#a6e3a1'),
        'N2_2': (3, 2, '12 * 6 = 72\nState: (4, 72)\nEval: impossible', '#f38ba8'),
        
        # Depth 2 (From N1_2)
        'N2_3': (5, 2, '20 + 7 = 27\nState: (6, 27)\nEval: maybe', '#f9e2af'),
        
        # Depth 3 (From N2_1)
        'N3_1': (1, 1, '12 * 2 = 24\nState: (24)\nEval: SUCCESS!', '#a6e3a1'),
        
        # Depth 3 (From N2_3)
        'N3_2': (5, 1, '27 - 6 = 21\nState: (21)\nEval: FAILED', '#f38ba8')
    }

    edges = [
        ('Root', 'N1_1'), ('Root', 'N1_2'), ('Root', 'N1_3'),
        ('N1_1', 'N2_1'), ('N1_1', 'N2_2'),
        ('N1_2', 'N2_3'),
        ('N2_1', 'N3_1'),
        ('N2_3', 'N3_2')
    ]

    G.add_edges_from(edges)
    pos = {node: (x, y) for node, (x, y, _, _) in nodes.items()}

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, arrows=True, arrowstyle='-|>', 
                           arrowsize=20, edge_color='#888888', width=2, connectionstyle='arc3,rad=0.0')

    # Draw nodes as rounded bounding boxes
    for node, (x, y, label, color) in nodes.items():
        # Draw background box (smaller width, slightly taller for bigger text)
        box = patches.FancyBboxPatch((x - 0.8, y - 0.35), 1.6, 0.7, 
                                     boxstyle="round,pad=0.1", 
                                     ec="#333333", fc=color, lw=1.5, alpha=0.9, zorder=2)
        ax.add_patch(box)
        
        # Text color
        text_color = 'black'
        font_weight = 'bold' if 'SUCCESS' in label or 'Input' in label else 'normal'
        
        # Larger font size
        ax.text(x, y, label, ha='center', va='center', fontsize=13, 
                color=text_color, fontweight=font_weight, zorder=3, family='sans-serif')

    # Styling
    ax.axis('off')
    ax.set_xlim(0, 9.5)
    ax.set_ylim(0, 5)

    # Legend
    legend_elements = [
        patches.Patch(facecolor='#a6e3a1', edgecolor='#333333', label='Promising Path (Sure)'),
        patches.Patch(facecolor='#f9e2af', edgecolor='#333333', label='Exploratory Path (Maybe)'),
        patches.Patch(facecolor='#f38ba8', edgecolor='#333333', label='Dead End (Impossible)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', frameon=False, 
              labelcolor='black', fontsize=14)

    plt.tight_layout()
    
    output_filename = 'tot_search_tree.png'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', transparent=True)
    print(f"Plot saved to: {output_filename}")

if __name__ == '__main__':
    create_tree_plot()
