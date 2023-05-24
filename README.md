## FeatEngine
#### Powerful Multi-Objective Optimization Engine for Feature Selection

* A light and general framework for feature selection.
* Some well implemented functions and algorithms for feature selection.

#### Algorithms

There are all implemented algorithms in this project:

- [x] CMDPSO-FS(OMOPSO)
- [x] Er-FS
- [x] NSPSO-FS
- [x] SM-MOEA-FS

#### Datasets

- [x] 9Tumor(**Features:** 5726, **Instance:** 60, **Class:** 9)

#### How to use?

1. Open `main.m`.
2. Configure the experimental parameters.

```matlab
%% Settings
N = 100; % Population size
tradeOff = 0.8; % Tradeoff factor
nfevalmax = 1E3; % Max function evaluation
runTimes = 30; % Total run times
```

3. Select one classification algorithm, such as `KNN: fitcknn`、`SVM: fitcsvm`...

```matlab
%% Classification Algorithm
CA = @fitcknn; 
```

4. Select one optimization Problem(dataset), you can find in `Problem` folder.

```matlab
%% Problem
Problem = NineTumor(nfevalmax);
```

5. Select the optimization algorithm, you can find in `Algorithms` folder.

```matlab
%% Evolutionary Computation
EC = @SMMOEAFS;
```

6. Run.

#### Support

- If you have more ideas about this project, you can express your opinion in **issues block** or directly submit your innovative version of **FeatEngine**.
- If you have any question, please contact with Kevin Kong([kevin@fanrenkong.com](mailto:kevin@fanrenkong.com)).
