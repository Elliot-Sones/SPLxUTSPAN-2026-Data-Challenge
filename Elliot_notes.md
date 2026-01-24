Creating more data: 

Can we create data that improves the models understanding with same algorithm and features set









was test 3 on all features done on all features in the training set? Why did you do test 5 on only wrist z-trajectory? all of this is just noise to me. I want to know what what tested how and the results. summarise this for everytest in one sentence and show a table of the results



Research: 

- Attention on the features? 



New testing plan: 

(possible features x possible models (+hyperparam) x shared vs per participant x per target output)


The Challenge                                                                                                                     
                                                                                                                                    
  - Winner margin: 0.000001                                                                                                         
  - Timeline: ~1 month                                                                                                              
  - Resource: CPU running 24/7                                                                                                      
  - Goal: Exhaust all possibilities until no improvement is possible                                                                
                                                                                                                                    
  Dimensions of the Search Space                                                                                                    
                                                                                                                                    
  Fixed dimensions (small):                                                                                                         
  - Training: 2 (shared, per-participant)                                                                                           
  - Target: 3 (angle, depth, left_right)                                                                                            
                                                                                                                                    
  Expandable dimensions (where the wins are):                                                                                       
                                                                                                                                    
  1. Features (potentially infinite)                                                                                                
                                                                                                                                    
  - Base sets: baseline, physics, hybrid, stats_full                                                                                
  - Subsets of each (feature selection)                                                                                             
  - Combinations (physics + subset of baseline)                                                                                     
  - Transformations (log, sqrt, polynomial)                                                                                         
  - Interaction terms (feature1 * feature2)                                                                                         
  - Rolling statistics at different windows                                                                                         
  - Different release frame detection methods                                                                                       
                                                                                                                                    
  2. Algorithms (dozens)                                                                                                            
                                                                                                                                    
  - Tree-based: LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees, HistGradientBoosting                                         
  - Linear: Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor                                                                 
  - Other: KNN, SVR (various kernels), MLP, GaussianProcess                                                                         
  - Each has 5-20 hyperparameters                                                                                                   
                                                                                                                                    
  3. Preprocessing                                                                                                                  
                                                                                                                                    
  - Scaling: None, Standard, MinMax, Robust, Quantile                                                                               
  - Outlier handling: clip, remove, winsorize                                                                                       
  - Missing value: median, mean, KNN impute                                                                                         
                                                                                                                                    
  4. Ensembling (meta-level)                                                                                                        
                                                                                                                                    
  - Simple average of top N models                                                                                                  
  - Weighted average (optimize weights)                                                                                             
  - Stacking with meta-learner                                                                                                      
  - Per-target ensembles                                                                                                            
                                                                                                                                    
  Strategy Questions                                                                                                                
                                                                                                                                    
  1. Search order - Breadth vs Depth?                                                                                               
  - Option A: Test all algorithms with default params first, then deep-dive winners                                                 
  - Option B: Deep-dive one algorithm at a time until exhausted                                                                     
  - Option C: Parallel tracks - breadth on algorithms, depth on features                                                            
                                                                                                                                    
  2. When is a dimension "exhausted"?                                                                                               
  - When improvement < 0.0001 for 100 consecutive experiments?                                                                      
  - When we've tested N% of the theoretical space?                                                                                  
  - When confidence interval overlaps with best?                                                                                    
                                                                                                                                    
  3. How to handle the combinatorial explosion?                                                                                     
  - Features alone: 40 physics features = 2^40 possible subsets                                                                     
  - Need smart pruning: importance-based, correlation-based                                                                         
                                                                                                                                    
  4. Reproducibility and precision                                                                                                  
  - Same seed across all experiments?                                                                                               
  - Multiple seeds and average?                                                                                                     
  - How many CV folds for final comparison?                                                                                         
                                                                                                                                    
  5. Tracking and analysis                                                                                                          
  - SQLite database for all experiments?                                                                                            
  - Automatic detection of promising directions?                                                                                    
  - Dashboard to monitor progress?                                                                                                  
                                                                                                                                    
  Proposed Exhaustive Search Framework                                                                                              
                                                                                                                                    
  Level 1: Coarse Grid (days 1-7)                                                                                                   
  - All algorithms x default params x all feature sets                                                                              
  - ~100 experiments                                                                                                                
  - Identify top 5 algorithms, top 2 feature sets                                                                                   
                                                                                                                                    
  Level 2: Hyperparameter Search (days 8-14)                                                                                        
  - Top 5 algorithms x extensive hyperparam grid                                                                                    
  - ~1000 experiments per algorithm                                                                                                 
  - Identify best config per algorithm                                                                                              
                                                                                                                                    
  Level 3: Feature Engineering (days 15-21)                                                                                         
  - Feature selection on best configs                                                                                               
  - Feature combinations                                                                                                            
  - New engineered features                                                                                                         
  - ~5000 experiments                                                                                                               
                                                                                                                                    
  Level 4: Ensembling (days 22-25)                                                                                                  
  - Combine top 20 models                                                                                                           
  - Optimize ensemble weights                                                                                                       
  - Stacking experiments                                                                                                            
  - ~500 experiments                                                                                                                
                                                                                                                                    
  Level 5: Fine-tuning (days 26-28)                                                                                                 
  - Decimal-level hyperparam tuning on winners                                                                                      
  - Multiple seed validation                                                                                                        
  - Statistical significance testing                                                                                                
  - ~1000 experiments                                                                                                               
                                                                                                                                    
  Buffer (days 29-31)                                                                                                               
  - Final validation                                                                                                                
  - Submission generation                                                                                                           
                                                                                                                                    
  Key Questions for You                                                                                                             
                                                                                                                                    
  1. Do you want to build an automated experiment runner that logs everything to a database and can run unattended?                 
  2. For the 0.000001 precision - should we run each promising config with 10+ different seeds to get stable estimates?             
  3. What's your compute budget? A month of CPU on vast.ai - what spec are you thinking? (affects parallelism)                      
  4. Do you want real-time monitoring/alerts when a new best is found?                                                              
                                                                                                                                    
  What aspects should we dig deeper into? 

