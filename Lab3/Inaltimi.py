import numpy as np

train_data = [(160, 'F'), (165, 'F'), (155, 'F'), (172, 'F'), (175, 'B'), (180, 'B'), (177, 'B'), (190, 'B')]
nr_all = len(train_data)

# total nr of males/females
nr_m = 0
nr_f = 0

# count boys/girls in each category
nr_m_g = np.zeros(5)
nr_f_g = np.zeros(5)
for person in train_data:
    height_group = (person[0] - 151) // 10
    if person[1] == 'F':
        nr_f += 1
        nr_f_g[height_group] += 1
    else:
        nr_m += 1
        nr_m_g[height_group] += 1

print ("Females: " + str(nr_f_g))
print ("Males: " + str(nr_m_g))


def predict(height):
    hg = (height - 151) // 10
    total_hg = nr_m_g[hg] + nr_f_g[hg]
    prob_hg = total_hg / nr_all

    # female
    if nr_f_g[hg] == 0:
        pred_fem = 0
    else:
        prob_f_hg = nr_f_g[hg] / total_hg
        prob_fem = (0.0 + nr_f) / nr_all
        pred_fem = (prob_hg * prob_f_hg) / prob_fem

    # male
    if nr_f_g[hg] == 0:
        pred_male = 0
    else:
        prob_m_hg = nr_m_g[hg] / total_hg
        prob_male = (0.0 + nr_m) / nr_all
        pred_male = (prob_hg * prob_m_hg) / prob_male

    if pred_fem > pred_male:
        return 'F'
    else:
        return 'M'


print(predict(198))
