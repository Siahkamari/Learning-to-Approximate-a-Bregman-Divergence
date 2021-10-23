import torch as th

class objectives():
  def __init__(self, rank=None, classify=None, is_similar=None):
    self.device = th.device("cuda:0" if th.cuda.is_available() else "cpu")
    self.rank = rank
    self.classify = classify
    self.is_similar = is_similar
  
  def mean_average_precision(self, y_que, X_que, y_pool, X_pool):
    if self.rank==None:
      raise("rank function is not provided.")
    n_que = len(y_que)
    n_pool = len(y_pool)
    ap = th.zeros(n_que, device=self.device)
    for i in range(n_que):
      y = y_que[i]
      ranks = self.rank(X_que[i], X_pool)
      rel = y_pool[ranks] == y
      pre_k = th.cumsum(rel, dim=0)/th.arange(1,n_pool+1, device=self.device)
      ap[i] = th.divide(th.sum(pre_k*rel), th.sum(rel))
    return th.nansum(ap)/(n_que-th.sum(th.isnan(ap)))

  def area_under_the_curve(self, y_que, X_que, y_pool, X_pool):
    if self.rank==None:
      raise("rank function is not provided.")
    n_que = len(y_que)
    auc = th.zeros(n_que, device=self.device)
    for i in range(n_que):
      y = y_que[i]
      ranks = self.rank(X_que[i], X_pool)
      y_count = th.sum(y_pool == y)
      swapped_pairs = th.sum( (y_pool[ranks]!=y) *(y_count -  th.cumsum(y_pool[ranks]==y, dim=0))  )
      auc[i] = 1 - swapped_pairs/(y_count*(len(y_pool)  -  y_count))
    return th.nansum(auc)/(n_que-th.sum(th.isnan(auc)))

  def pairwise_score(self, y_que, X_que):
    if self.is_similar == None:
      raise("is_similar function is not provided.")
    n_que = len(y_que)
    score = th.zeros(n_que, device=self.device)
    for i in range(n_que):
      iota = y_que[i] == y_que
      iota_pred = self.is_similar(X_que[i], X_que)
      score[i] = th.mean((iota == iota_pred).float())
    return th.mean(score)
  
  def accuracy(self, y_que, X_que):
    if self.classify == None:
      raise("classify function is not provided.")
    y_pred = self.classify(X_que)
    acc = th.mean((y_que==y_pred).float())
    return acc