from math import *
from decimal import Decimal

class Benzerlik():
    
    def oklit(self, x, y):
        return sqrt(sum(pow(a-b,2) for a,b in zip(x,y)))
    
    def manhattan(self, x, y):
        return sum(abs(a-b) for a,b in zip(x,y))
    
    def minkowski(self, x, y, p):
        return self.nth_root(sum(pow(abs(a-b),p) for a,b in zip(x,y)),p)
        
    def nth_root(self, value, n_root):
        root_value = 1/float(n_root)
        return round(Decimal(value)** Decimal(root_value),3)
    
    def kosinus(self, x, y):
        pay = sum(a*b for a,b in zip(x,y))
        payda = self.kare_kok(x)*self.kare_kok(y)
        return round(pay/float(payda),3)
    
    def kare_kok(self,x):
        return round(sqrt(sum(a*a for a in x)),3)
    