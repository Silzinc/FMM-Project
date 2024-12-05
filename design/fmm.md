[toc]

## Jonas

$$
\def\x{\vec x}
\def\z{\vec z}
\def\d{\mathrm d}
$$

### Étapes pour l'algorithme FMM

#### Initialisation

1. Choisir un volume de simulation et un nombre de particules (cube de taille $R$).
2. Déterminer $\phi$.

   1. Choisir une fonction $\eta$ d'adoucissement pour le potentiel des particules massives.
   2. Se donner une largeur de particule $\varepsilon$ (typiquement $4R/\sqrt N$ avec $R$ la taille du volume de simulation et $N$ le nombre de particules).
   3. En déduire la fonction $\varphi$ telle que le potentiel en $\x$ s'écrive
      $$
      \Phi(\x) = -\sum_{i=1}^N \dfrac{G\mu_i}{\varepsilon} \varphi\left(\dfrac{\|\x - \x_i\|}{\varepsilon}\right)
      $$
      Que l'on obtient via l'équation de Gauss gravitationnelle :
      $$
       4\pi \eta (\xi) = -\dfrac 1 {\xi ^2} \frac{\d }{\d \xi}\left(\xi^2 \frac{\d \varphi}{\d \xi}\right)
      $$
      Un choix typique est l'adoucissement de Plummer
      $$
      \eta (\xi) = \frac 3 {4\pi} (1 + \xi^2)^{-5/2}\\
      \varphi(\xi) = (1 + \xi^2)^{-1/2}
      $$
   4. En déduire $\phi(\vec r) = -\dfrac G \varepsilon \varphi\left(\dfrac{\|\vec r\|}{\varepsilon}\right)$.
   5. Calculer le gradient et éventuellement la Hessienne de $\phi$ analytiquement. Pour une fonction radiale $F(\vec r) = f(r/\varepsilon)$, le gradient et les coefficients de la Hessienne s'écrivent par (avec $\vec \xi = \vec r/\varepsilon$):

      $$
      \begin{aligned}
      \vec \nabla F(\vec r) &= \frac{\vec \xi}{\varepsilon \xi} f'\left(\xi\right) = \frac{\vec r}{r^2} \xi f'(\xi)\\
      \frac{\partial ^2 F(\vec r)}{\partial x_i \partial x_j} &= \frac 1 {\varepsilon^2 \xi} \left(\delta_{ij} f'(\xi) + \frac{\xi_i \xi_j}{\xi} \left(f''(\xi) - \frac{f'(\xi)} \xi \right)\right) = \frac 1 {r^2} \left(\xi\delta_{ij} f'(\xi) + \xi_i \xi_j \left(f''(\xi) - \frac{f'(\xi)} \xi \right)\right)
      \end{aligned}
      $$

      Dans le cas de l'adoucissement de Plummer,

      $$
      \begin{aligned}
      \varphi(\xi) &= \frac 1 {\sqrt{1 + \xi^2}}\\
      \phi(\vec r) &= -\frac G r \underbrace{\frac \xi {\sqrt{1 + \xi^2}}}_{\text{Fonction } \texttt{phi}}\\
      \varphi'(\xi) &= -\frac{\xi}{(1 + \xi^2)^{3/2}}\\
      \vec \nabla \phi(\vec r) &= -\frac{G}{r^2} \times \underbrace{\left(-\frac{\xi^2}{(1 + \xi^2)^{3/2}}\vec \xi\right)}_{\text{Fonction } \texttt{grad\_phi}}\\
      \varphi''(\xi) &= \frac{2\xi^2 - 1}{(1 + \xi^2)^{5/2}}\\
      \frac{\partial ^2 \phi(\vec r)}{\partial x_i \partial x_j} &= -\frac G {r^3} \underbrace{\xi^3\left(\frac{3\xi_i \xi_j }{(1 + \xi^2)^{5/2}}-\frac{\delta_{ij}}{(1 + \xi^2)^{3/2}}\right)}_{\text{Fonction } \texttt{hess\_phi}}
      \end{aligned}
      $$

3. Choisir des masses et des positions pour les particules.
4. Construire l'arbre des multipoles.
   1. Choisir le nombre maximal $n_{max}$ de particules dans une cellule feuille.
   2. En prenant le volume entier comme racine, donner récursivement à chaque noeud 8 cellules filles (en coupant en deux dans chaque dimension) tant que le nombre de particules dans le noeud est strictement supérieur $n_{max}$.
   3. En partant des cellules feuilles, dans une cellule $A$, calculer la masse $M_A$ de la cellule (somme des masses des particules), son barycentre $\z_A$ et sa taille caractéristique $w_A = \max_{x\in A}\{\|\x - \z_A\|\}$.
   4. Remonter ensuite la récursion de l'arbre. Pour une cellule mère $A$ avec des cellules filles $(B_i)$, on a $M_A = \sum_{i=1}^8 M_{B_i}$, $M_A\z_A = \sum_{i=1}^8 M_{B_i}\z_{B_i}$ et $w_A = \max_{1\le i \le 8}\{\|\z_A - \z_{B_i}\| + w_{B_i}\}$.
5. Calculer les tenseurs de champs des cellules.
   1. Ces tenseurs ont 4 coordonnées. Les initialiser à $(0, 0, 0, 0)$ pour toutes les cellules (peut être fait dans l'étape 4).
   2. À suivre... (pas encore trouvé comment implémenter efficacement la descente dans l'arbre pour le calcul des champs).

```python
class OctTree:
   """
   Fields:
      children: children of the node (list of 8 OctTree | None)
      dtype: type of the node (Any)
      value: value of the node (dtype)
   """

class FMMCell:
   """
   Fields:
      width: max distance between barycenter and contained particles (float)
      samples: list of mass samples in the cell, None if not leaf (List[MassSample] | None)
      barycenter: centre of mass (vec3)
      mass: mass of the cell (float)
      field_tensor: when leaf cell sum of field tensor contributions of all other far cells (vec4)
      neighbors: list of neighboring cells (List[FMMCell])

   """

class MassSample:
   """
   Fields:
      mass: mass of the particle (float)
      position: position of the particle (vec3)
      previous_position: previous position of the particle (for Verlet integration) (vec3)
   Methods:
      speed: speed of the particle (float -> vec3)
   """

class FMMSolver:
   """
   Fields:
      size: size of the simulation cube (float)
      gradphi: function of r/epsilon representing the gradient
               potential of one "particle" (Callable[[float], float])
      (hess_phi: function of r/epsilon giving the structure of
                 the Hessian matrix of the potential of one "particle"
                 (Callable[[float], matrix3x3]))
      dt: timestep (float)
   Methods:
      epsilon: particle smoothing size (None -> float)

   """
```
