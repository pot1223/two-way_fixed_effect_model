import pandas as pd 
import statsmodels.api as sm  

data = pd.read_csv('/ab_dataDeMean.csv')

# demW_log1p_pageviews = y[it] - avg(y[t])
# demF_log1p_pageviews = y[it] - avg(y[i])
# demWF_log1p_pageviews = y[it] - avg(y[i]) - avg(y[t]) + avg(y)

# demW_using_ab_only = ABTesting[it] - ABTesting[t]   
# demF_using_ab_only = ABTesting[it] - ABTesting[i]   
# demF_log1p_stack2 = TechStack[it] - TechStack[i]   
# demWF_using_ab_only = ABTesting[it] − avg(ABTesting[i]) − avg(ABTesting[t]) + avg(ABTesting)
# demWF_log1p_stack2 = TechStack[it] - avg(TechStack[i]) - avg(TechStack[t]) + avg(TechStack)

# 기업의 시간 고정효과만 통제한 모델 
X = sm.add_constant(data[['demW_using_ab_only']]) 
X = pd.concat([X], axis=1)
ols_model = sm.OLS(data['demW_log1p_pageviews'], X) 
ols_results = ols_model.fit(cov_type='cluster', cov_kwds={'groups': data['firm_id']})
print(ols_results.summary()) 

# AB test 도입 시 방문자가 약 218% 증가하는 효과를 보인다 

# 기업 고정효과를 통제한 모델
X = sm.add_constant(data[['demWF_using_ab_only']]) 
X = pd.concat([X], axis=1)
ols_model = sm.OLS(data['demWF_log1p_pageviews'], X) 
ols_results = ols_model.fit(cov_type='cluster', cov_kwds={'groups': data['firm_id']})
print(ols_results.summary()) 

# 기업 고정효과를 통제한 AB test 도입 시 방문자가 약 70% 증가하는 효과를 보인다 

# 기업의 고정효과와 기술스택을 통제한 모델
X = sm.add_constant(data[['demWF_using_ab_only','demWF_log1p_stack2']]) 
X = pd.concat([X], axis=1)
ols_model = sm.OLS(data['demWF_log1p_pageviews'], X) 
ols_results = ols_model.fit(cov_type='cluster', cov_kwds={'groups': data['firm_id']})
print(ols_results.summary()) 

# 기업 고정효과와 기존 기술스택을을 통제한 AB test 도입 시 방문자가 약 1.6% 증가하는 효과를 보인다
