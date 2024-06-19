import numpy as np

# Data for accuracy and loss for each user across epochs
data = [
    [
        {0: 0.05945945945945946, 1: 0.05945945945945946, 2: 0.05945945945945946}, 
        {0: 0.2117117117117117, 1: 0.21857120278172915, 2: 0.2179892220988112}, 
        {0: -0.004504504504504499, 1: -0.004504504504504499, 2: -0.004504504504504499}, 
        {0: -0.004504504504504499, 1: -0.004504504504504499, 2: -0.004504504504504499}
    ], 
    [
        {0: 0.4716849662162162, 1: 0.4716849662162162, 2: 0.4716849662162162}, 
        {0: -0.318312697407127, 1: -0.31629088841777875, 2: -0.31479583193192745}, 
        {0: 0.022623040296651863, 1: 0.022623018616186267, 2: 0.022623036549733282}, 
        {0: 0.03103133270369597, 1: 0.031031343817280243, 2: 0.03103136788286927}
    ]
]

# Calculate cumulative sum of accuracy and loss for each user
cumsum_results = {}
for user_id in data[0][0].keys():  # assuming all dicts have the same user keys
    cumsum_results[user_id] = {
        'accuracy_cumsum': np.cumsum([epoch[user_id] for epoch in data[0]]),
        'loss_cumsum': np.cumsum([epoch[user_id] for epoch in data[1]])
    }

print(cumsum_results)
'''
{0: {'accuracy_cumsum': array([0.05945946, 0.27117117, 0.26666667, 0.26216216]), 
     'loss_cumsum': array([0.47168497, 0.15337227, 0.17599531, 0.20702664])}, 
 1: {'accuracy_cumsum': array([0.05945946, 0.27803066, 0.27352616, 0.26902165]), 
     'loss_cumsum': array([0.47168497, 0.15539408, 0.1780171 , 0.20904844])}, 
 2: {'accuracy_cumsum': array([0.05945946, 0.27744868, 0.27294418, 0.26843967]), 
     'loss_cumsum': array([0.47168497, 0.15688913, 0.17951217, 0.21054354])}
}
'''
print(cumsum_results[0]) #For user 0
'''
{
'accuracy_cumsum': array([0.05945946, 0.27117117, 0.26666667, 0.26216216]), 
'loss_cumsum': array([0.47168497, 0.15337227, 0.17599531, 0.20702664])
}
'''
print(cumsum_results[0]['accuracy_cumsum']) #accuracy cumsum for user 0
'''
[0.05945946 0.27117117 0.26666667 0.26216216]
'''
print(cumsum_results[0]['accuracy_cumsum'][0]) #accuracy cumsum for user 0 at epoch 0
'''
0.05945945945945946
'''
