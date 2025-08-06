# cudf를 이용한 k-mean

# 필요한것들 improt 해주고,
import cudf
import cuml

import cuxfilter as cxf

#data load 해보겠습니다

gdf = cudf.read_csv('./data/clean_uk_pop.csv', usecols=['easting', 'northing'])
print(gdf.dtypes)
gdf.shape

#k-means Clustering
# instantaite
km = cuml.KMeans(n_clusters=5) # 5개의 그룹으로 나누고,

# fit
km.fit(gdf) # 모델을 훈련시키기 !

# assign cluster as new column
gdf['cluster'] = km.labels_
km.cluster_centers_

## 어떻게 클러스터 됐는지 눈으로 확인해볼까?
# associate a data source with cuXfilter
cxf_data = cxf.DataFrame.from_dataframe(gdf)

# define charts
scatter_chart = cxf.charts.datashader.scatter(x='easting', y='northing')

# define widget using the `cluster` column for multiselect
# use the same technique to scale the scatterplot, then add a widget to let us select which cluster to look at
cluster_widget = cxf.charts.panel_widgets.multi_select('cluster')

# create dashboard
dash = cxf_data.dashboard(charts=[scatter_chart],sidebar=[cluster_widget], theme=cxf.themes.dark, data_size_widget=True)

dash.app()

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)


### dbscan입니다! eps거리 안에 point가 있으면 하나의 그룹
# 장점은 이상치를 찾을 수 있다는것?

gdf = cudf.read_csv('./data/pop_sample.csv', dtype=['float32', 'float32', 'float32'])
print(gdf.dtypes)
gdf.shape
#예시데이터는 감염의 여부를 나타내는 데이터인거 같네욤

# dbscan은 클러스터 수를 미리 알 수 없고, 클러스터가 오목하거나 기타 특이한
#모양을 가질 수 있을때 굉장히 효과적

dbscan = cuml.DBSCAN(eps=5000)
# dbscan = cuml.DBSCAN(eps=10000)

infected_df = gdf[gdf['infected'] == 1].reset_index()
infected_df['cluster'] = dbscan.fit_predict(infected_df[['northing', 'easting']])
infected_df['cluster'].nunique()

# 핵심은 거리, 거리안에 있으면 하나의 클러스트로 밀집으로 묶임
#eps를 정해줘야됨

# eps가 10,000이라면?
dbscan = cuml.DBSCAN(eps=10000)
infected_df = gdf[gdf['infected'] == 1].reset_index()
infected_df['cluster'] = dbscan.fit_predict(infected_df[['northing', 'easting']])
infected_df['cluster'].nunique()

# 시각화를 한다면??
infected_df.to_pandas().plot(kind='scatter', x='easting', y='northing', c='cluster')

import IPython
app = IPython.Application.instance()
app.kernel.do_shutdown(True)
