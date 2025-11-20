import matplotlib.pyplot as plt
import numpy as np

fragments = list(range(1, 33))
tacts = [2, 2, 2, 2, 2, 2, 2, 3, 9, 8, 9, 10, 11, 11, 11, 11,
         11, 11, 11, 11, 11, 11, 12, 12, 27, 34, 38, 41, 52, 57, 63, 65]

plt.figure(figsize=(14, 8))

plt.plot(fragments, tacts, 'bo-', linewidth=2, markersize=6, label='Время доступа')

plt.title('Зависимость времени доступа к элементу от числа фрагментов',
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Число фрагментов', fontsize=12)
plt.ylabel('Время доступа (такты)', fontsize=12)
plt.grid(True, alpha=0.3)

plt.xticks(fragments, fontsize=9)

plt.axvline(x=8, color='red', linestyle='--', alpha=0.5, label='L1: 8')
plt.axvline(x=11, color='orange', linestyle='--', alpha=0.5, label='L2: 11')
plt.axvline(x=24, color='green', linestyle='--', alpha=0.5, label='L3: 24')

plt.legend()
plt.tight_layout()

plt.savefig('graphik.png', dpi=300, bbox_inches='tight')
plt.show()
