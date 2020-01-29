__author__ = "Fedyushin Ilya"
__email__  = "Fedushin.ilya@yandex.ru | hechachi1997@gmail.com"


"""
Ниже представлен алгоритм диф. эволюции, работающий любыми ограничениями на переменные.
Идея для реализация взята в учебнике [1].
Также представлен пример использования.  

[1] https://pdfs.semanticscholar.org/96de/70af9d9e7da5cdf577c6fbeb7f1a90491efc.pdf
[2] https://pablormier.github.io/2017/09/05/a-tutorial-on-differential-evolution-with-python/

"""

import numpy as np
from scipy.optimize import rosen


def de(fobj, args, bounds, constrain_func_list, R, seed, mut=0.8, crossp=0.7, popsize=20, its=1000):
    """
    Реализация алгоритма дифференциальной эволюции для задачи глобальной оптимизации (минимизации) с линейными ограничениями.
    
    --------
    Аргументы:
        fobj (object) : минимизируемая функция.
        args (tuple) : параметры, которые передуются функции fobj.
        bounds (list of tuples) : список (min_i, max_i) на оптимизируемые переменные вида min_i <= x_i <= max_i.
        constrain_func_list (list of object) : список функций для ограничений типа неравенства (<=).
        R (list of float) : коэффициент регуляризации.
        seed (int) : random seed.
        mut (float) : коэффициент мутации.
        crossp (float) : коэффициент наследования.
        popsize (float) : размер популяции.
        its (int) : количество итераций.
    """
    
    # размерность пространства популяции.
    dimensions = len(bounds)
    
    # размер популяции.
    actual_popsize = popsize
    
    # нормированная популяция.
    np.random.seed(seed)
    pop = np.random.rand(actual_popsize, dimensions)
    
    min_b, max_b = np.asarray(bounds).T
    
    diff = np.fabs(min_b - max_b)
    
    # реальная популяция.
    pop_denorm = min_b + pop * diff
    
    # значение функции.
    fitness = np.asarray([fobj(elem, *args) for elem in pop_denorm])
    
    # регуляризация 1 (штрафуем obj при невыполнении ограничений)
    regularization_1 = np.asarray([
        sum((R*max(0, diff_func['fun'](elem)) for idx, diff_func in enumerate(constrain_func_list)))
        for elem in pop_denorm
    ])
    
    # для каждого элемента популяции: лежит ли в ОДЗ?
    feasible = np.asarray([elem == 0 for elem in regularization_1])
    
    # значение alpha.
    alpha = 0
    if np.any(feasible) and np.any(~feasible):
        alpha = max(0, np.max(fitness[feasible]) - np.min(fitness[~feasible]))
    
    # регуляризация 2 (ходим, чтобы лучший элемент не из ОДЗ был хуже худшего из ОДЗ)
    regularization_2 = np.zeros_like(fitness)
    for idx, elem in enumerate(feasible):
        if not np.any(feasible) or elem == 0:
            regularization_2[idx] = 0
        if np.any(feasible) and (elem > 0):
            regularization_2[idx] = alpha
        
    fitness = fitness + regularization_1 + regularization_2
    
    best_idx = np.argmin(fitness)
    best = pop_denorm[best_idx]
    
    for i in range(its):
        
        alpha = 0
        if np.any(feasible) and np.any(~feasible):
            alpha = max(0, np.max(fitness[feasible]) - np.min(fitness[~feasible]))
            
        for j in range(actual_popsize):
            idxs = [idx for idx in range(actual_popsize) if idx != j]
            a, b, c = pop[np.random.choice(idxs, 3, replace=False)]
            mutant = np.clip(a + mut * (b - c), 0, 1)
            cross_points = np.random.rand(dimensions) < crossp
            if not np.any(cross_points):
                cross_points[np.random.randint(0, dimensions)] = True
            trial = np.where(cross_points, mutant, pop[j])
            trial_denorm = min_b + trial * diff
            
            reg = sum((R*max(0, diff_func['fun'](trial_denorm)) 
                        for idx, diff_func in enumerate(constrain_func_list)))
            
            if (not np.any(feasible)) or (reg == 0):
                teta = 0
            if np.any(feasible) and (reg > 0):
                teta = alpha
                
            f = fobj(trial_denorm, *args) + reg + teta
            if f < fitness[j]:
                fitness[j] = f
                pop[j] = trial
                feasible[j] = reg == 0
                if f < fitness[best_idx]:
                    best_idx = j
                    best = trial_denorm
                          
        yield best, fitness[best_idx]

def main(*args, **kwargs):

    # ограничения сверху и снизу на каждую переменную.
    bounds = [(0, 2), (0, 2), (0, 2), (0, 2), (0, 2)]

    # ограничение суммы переменных.
    constrain_func_list = []
    def f_1(x, sum_boundary=1.0):
        return np.sum(x) - sum_boundary
    
    constrain_func_list.append({
        'type': 'ineq',
        'fun': f_1
    })

    result = de(
        fobj=rosen,
        args=(),
        bounds=bounds,
        constrain_func_list=constrain_func_list,
        popsize=50,
        seed=0,
        mut=0.7,
        crossp=0.9,
        its=1000,
        R=10.0)

    result_array = np.asarray(list(result))

    print("Значения переменных: {}".format(result_array[-1][0]))
    print("Сумма значений переменных: {}".format(sum(result_array[-1][0])))
    print("Оптимальное значение: {}".format(result_array[-1][1]))

    return 0

if __name__ == '__main__':
    main()    