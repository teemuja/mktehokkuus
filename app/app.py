import streamlit as st
import pandas as pd
import geopandas as gpd
import numpy as np
import h3
from pathlib import Path
from shapely.geometry import mapping, shape
import geocoder
from owslib.wfs import WebFeatureService
from owslib.fes import PropertyIsLike
from owslib.etree import etree
import plotly.express as px
import plotly.graph_objects as go
px.set_mapbox_access_token(st.secrets['MAPBOX_TOKEN'])
mbtoken = st.secrets['MAPBOX_TOKEN']
my_style = st.secrets['MAPBOX_STYLE']

st.set_page_config(page_title="Research App", layout="wide", initial_sidebar_state='expanded')
st.markdown("""
<style>
button[title="View fullscreen"]{
        visibility: hidden;}
</style>
""", unsafe_allow_html=True)

#title
st.header("Väestötiheyden kehitys Suomessa",divider='green')

# content
path = Path(__file__).parent / 'data/kunta_dict.csv'
with path.open() as f1:
    kuntakoodit = pd.read_csv(f1, index_col=False, header=0).astype(str)
kuntakoodit['koodi'] = kuntakoodit['koodi'].str.zfill(3)
kuntalista = kuntakoodit['kunta'].tolist()
#st.title(':point_down:')
st.subheader('Tällä tutkimusappilla voit katsoa, miten seudun väestötiheys on kehittynyt.')
# kuntavalitsin
k1,k2 = st.columns([2,1])
valinnat = k1.multiselect('Valitse kunnat (max 7) - kattavuus koko Suomi', kuntalista, default=['Helsinki','Espoo','Vantaa'])
plot_mode = k2.radio('Väestöryhmä',('vaesto','ika_0_14','ika_65_'),horizontal=True)
st.caption('Ensin valittua käytetään väestögradientin keskipisteenä.')
vuodet = st.slider('Aseta aikajakso',2010, 2025, (2020, 2025),step=1)
#st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

#statgrid change
@st.cache_data()
def muutos_h3(kunta_list,y1=2020,y2=2025):
    url = 'http://geo.stat.fi/geoserver/vaestoruutu/wfs'
    wfs11 = WebFeatureService(url=url, version='1.1.0')
    path = Path(__file__).parent / 'data/kunta_dict.csv'
    with path.open() as f1:
        kuntakoodit = pd.read_csv(f1, index_col=False, header=0).astype(str)
    kuntakoodit['koodi'] = kuntakoodit['koodi'].str.zfill(3)
    kunta_dict_inv = pd.Series(kuntakoodit.koodi.values, index=kuntakoodit.kunta).to_dict()
    cols = ['grd_id','kunta','vaesto','ika_0_14','ika_65_','geometry']
    yrs = [y1,y2]
    # loop
    grid = pd.DataFrame()
    for kunta in kunta_list:
        koodi = kunta_dict_inv.get(kunta)
        filter = PropertyIsLike(propertyname='kunta', literal=koodi, wildCard='*')
        filterxml = etree.tostring(filter.toXML()).decode("utf-8")
        grid_kunta = pd.DataFrame()
        for y in yrs:
            response = wfs11.getfeature(typename=f'vaestoruutu:vaki{y}_1km_kp', filter=filterxml, outputFormat='json')
            griddata = gpd.read_file(response)[cols]
            griddata['vuosi'] = y
            grid_kunta = pd.concat([grid_kunta,griddata], ignore_index=True)
        grid = pd.concat([grid,grid_kunta], ignore_index=True)
    # yr pop
    grid.loc[grid['vuosi'] == y1,f'{y1}_tot'] = grid['vaesto']
    grid.loc[grid['vuosi'] == y2,f'{y2}_tot'] = grid['vaesto']
    # yr pop_lap
    grid.loc[grid['vuosi'] == y1,f'{y1}_lap'] = grid['ika_0_14']
    grid.loc[grid['vuosi'] == y2,f'{y2}_lap'] = grid['ika_0_14']
    # yr pop_van
    grid.loc[grid['vuosi'] == y1,f'{y1}_van'] = grid['ika_65_']
    grid.loc[grid['vuosi'] == y2,f'{y2}_van'] = grid['ika_65_']
    # prepare merge sum
    grid.replace({-1:0}, inplace=True)
    grid.loc[grid['vuosi'] == y1,'vaesto'] = -abs(grid['vaesto']) # make first yr value negative for merge sums below
    grid.loc[grid['vuosi'] == y1,'ika_0_14'] = -abs(grid['ika_0_14'])
    grid.loc[grid['vuosi'] == y1,'ika_65_'] = -abs(grid['ika_65_'])
    # count change with groupby..
    sums = grid.drop(columns='geometry').groupby(by='grd_id').sum().reset_index()
    sums_df = pd.merge(sums,grid[['grd_id','geometry']],on='grd_id')
    
    # create h3
    def assing_h3_and_polygons(df_in,reso=8):
        df = df_in.copy()
        gdf = gpd.GeoDataFrame(df,geometry='geometry',crs=3067).to_crs(4326)
        gdf['lng'] = gdf.geometry.x
        gdf['lat'] = gdf.geometry.y
        #put id in df
        df['h3_id'] = gdf.apply(lambda row: h3.latlng_to_cell(row['lat'], row['lng'], res=reso), axis=1)
        #gen cell polygon geometry
        df['geojsonpolygon'] = df['h3_id'].apply(lambda cell: h3.cells_to_geo([cell], tight=True))
        df['geometry'] = df['geojsonpolygon'].apply(lambda p: shape(p))
        del gdf
        gdf_out = gpd.GeoDataFrame(df, geometry='geometry')
        del df
        return gdf_out.drop(columns='geojsonpolygon')
    
    h3_out = assing_h3_and_polygons(sums_df) #gdf
 
    # count ratios of change
    h3_out['vaestosuht'] = round((h3_out['vaesto'] / h3_out[f'{y1}_tot'])*100,0)
    h3_out['ika_0_14suht'] = round((h3_out['ika_0_14'] / h3_out[f'{y1}_lap'])*100,0)
    h3_out['ika_65_suht'] = round((h3_out['ika_65_'] / h3_out[f'{y1}_van'])*100,0)

    return h3_out

def binit(df,color_col,bin_labels):
    min1 = df.loc[df[color_col] < 0][color_col].quantile(0.75)
    min2 = df.loc[df[color_col] < 0][color_col].quantile(0.25)
    max1 = df.loc[df[color_col] > 0][color_col].quantile(0.25)
    max2 = df.loc[df[color_col] > 0][color_col].quantile(0.75)
    top = df.loc[df[color_col] > 0][color_col].quantile(0.99)
    df['Muutos'] = pd.cut(x=df[color_col],bins=[-np.inf,min2,min1,max1,max2,top,np.inf],labels=bin_labels)
    df['keep'] = df['Muutos'].notnull()
    df = df.sort_values('keep', ascending=False).drop_duplicates(subset=['h3_id'])
    df = df.drop(columns=['keep'])
    df_out = df.loc[df['Muutos'] != 'ei muutosta']  #[(df[color_col] < -5) | (df[color_col] > 5)]
    df_out['Muutos'] = df_out['Muutos'].cat.remove_unused_categories()
    return df_out

def generate_plot_df(df_in,plot_mode):
    
    #bin it
    bin_labels = ['taantumaa','hiipumaa','ei muutosta','karttumaa','kasvua','top']
    plot = binit(df_in,plot_mode,bin_labels)
    
    #plot_mode
    if plot_mode == 'vaesto':
        graph_value = 'tot'
        value_title = 'kaikki ikäluokat'
    elif plot_mode == 'ika_0_14':
        graph_value = 'lap'
        value_title = 'lapset 0-14v'
    else:
        graph_value = 'van'
        value_title = 'väestö yli 65v'

    plot = plot.reset_index(drop=True)
    plot['feature_id'] = plot.index.astype(str)

    geojson = {
        'type': 'FeatureCollection',
        'features': [
            {
                'type': 'Feature',
                'id': row.feature_id,
                'properties': {},
                'geometry': mapping(row.geometry),
            }
            for row in plot.itertuples(index=False)
        ],
    }
    
    #colors
    bin_colors = {
        'taantumaa':'cornflowerblue',
        'hiipumaa':'lightblue',
        'ei muutosta':'ghostwhite',
        'karttumaa':'burlywood',
        'kasvua':'brown',
        'top':'red',
    }
    muutos_orders = [label for label in bin_labels if label in plot['Muutos'].dropna().unique()]

    # plot
    lat = plot.unary_union.centroid.y
    lng = plot.unary_union.centroid.x
    fig = px.choropleth_mapbox(plot,
                                geojson=geojson,
                                locations='feature_id',
                                featureidkey='id',
                                title=f'Väestötiheyden muutos {vuodet[0]}-{vuodet[1]}, ({value_title})',
                                color='Muutos',
                                hover_data=['vaesto','ika_0_14','ika_65_','vaestosuht','ika_0_14suht','ika_65_suht'],
                                center={"lat": lat, "lon": lng},
                                mapbox_style=my_style,
                                color_discrete_map=bin_colors,
                                category_orders={'Muutos': muutos_orders},
                                labels={'vaesto':'Muutos','ika_0_14':'Muutos lapset','ika_65_':'Muutos vanh.',
                                        'vaestosuht':'Muutos%','ika_0_14suht':'Muutos% lapset','ika_65_suht':'Muutos% vanh.'
                                        },
                                zoom=9,
                                opacity=0.5,
                                width=1200,
                                height=700
                                )
    fig.update_layout(margin={"r": 10, "t": 50, "l": 10, "b": 10}, height=700,
                                legend=dict(
                                    yanchor="top",
                                    y=0.97,
                                    xanchor="left",
                                    x=0.02
                                )
                                )

    return plot, fig, graph_value, value_title



# MAP HERE
with st.container() #st.expander('Kasvu kartalla', expanded=False):
    mapholder = st.empty()

#selectors
if len(valinnat) == 0:
    st.warning('Valitse kunnat')
    st.stop()
elif len(valinnat) > 7:
    st.warning('max 7')
    st.stop()
else:
    # generate regional data
    growth_df = muutos_h3(valinnat, y1=vuodet[0], y2=vuodet[1])
    plot, fig, graph_value, value_title = generate_plot_df(growth_df,plot_mode)
    mapholder.plotly_chart(fig, use_container_width=True)

#st.markdown('---')
# graph placeholder
#st.subheader('Väestögradientti')
den_holder = st.empty()

def den_grad(df_in,center_add,value,reso=8,rings=7):
    #put h3 as index
    h3_df = df_in.set_index('h3_id')
    # create center hex
    loc = geocoder.mapbox(center_add, key=mbtoken)
    h3_center = h3.latlng_to_cell(loc.lat,loc.lng, res=reso)
    
    # create grad_df to sum medians from the rings
    grad_df = pd.DataFrame()
    grads = []
    popsums = []
    # create ring columns around h3_center
    for i in range(1,rings+1):
        ring = pd.DataFrame()
        ring['h3_id'] = h3.grid_disk(h3_center, i)
        ring[value] = 0 #np.NaN
        ring.set_index('h3_id', inplace=True)
        ring[value].update(h3_df[value])
        # remove zeros
        ring = ring.loc[ring[value] != 0]
        popmedian = ring[value].median()
        popsum = ring[value].sum()
        grads.append(popmedian)
        popsums.append(popsum)
    grad_df['pop_median_5km2'] = grads
    grad_df['pop_sum_ring'] = popsums
    # create ring names
    grad_df.reset_index(drop=False, inplace=True)
    grad_df.rename(columns={'index':'ring'}, inplace=True)
    grad_df['ring'] = grad_df['ring'].astype(str) + ' km'
    return grad_df


# and density gradients + rings ..broken. fix someday...
#den0 = den_grad(df_in=plot,center_add=valinnat[0],value=f'{vuodet[0]}_{graph_value}',reso=8,rings=16)
#den1 = den_grad(df_in=plot,center_add=valinnat[0],value=f'{vuodet[1]}_{graph_value}',reso=8,rings=16)

#den0['pop_per_ring'] = den0['pop_sum_ring'].diff().fillna(den0['pop_sum_ring'])
#den1['pop_per_ring'] = den1['pop_sum_ring'].diff().fillna(den1['pop_sum_ring'])

# graph plotter
def generate_den_graphs(den0,den1):
    
    def plot_muutos(df1, df2):

        fig = go.Figure(
            layout=dict(
                title=f"Luokan '{value_title}' tiheys (~km² mediaani) & väestömäärä etäisyysvyöhykkeillä kohteesta '{valinnat[0]}'",
                yaxis=dict(
                    title="Tiheys (~km² mediaani)",
                    side="left",
                ),
                yaxis2=dict(
                    title="Väestömäärä etäisyysvyöhykkeellä",
                    overlaying="y",
                    side="right",
                )
            )
        )

        # Density (left y-axis)
        fig.add_trace(go.Scatter(x=df1['ring'], y=df1['pop_median_5km2'], name=f'{vuodet[0]} (Tiheys)',
                                fill='tozeroy', fillcolor='rgba(222, 184, 135, 0.5)',
                                mode='lines', line=dict(width=0.5, color='black'),
                                yaxis="y1"))

        fig.add_trace(go.Scatter(x=df2['ring'], y=df2['pop_median_5km2'], name=f'{vuodet[1]} (Tiheys)',
                                fill='tonexty', fillcolor='rgba(205, 127, 50, 0.5)',
                                mode='none', yaxis="y1"))

        # Total Population (right y-axis)
        fig.add_trace(go.Scatter(x=df1['ring'], y=df1['pop_per_ring'], name=f'{vuodet[0]} (Väestömäärä)',
                                mode='lines', line=dict(width=1.5, color='grey'),
                                yaxis="y2"))

        fig.add_trace(go.Scatter(x=df2['ring'], y=df2['pop_per_ring'], name=f'{vuodet[1]} (Väestömäärä)',
                                mode='lines', line=dict(width=1.5, color='black'),
                                yaxis="y2"))
        
        # Update Axes
        fig.update_xaxes(range=[1, 15])
        fig.update_layout(margin={"r": 10, "t": 50, "l": 10, "b": 10}, height=500,
                        legend=dict(yanchor="top", y=0.97, xanchor="right", x=0.99))

        # Add a vertical dashed line at x=3
        fig.update_layout(shapes=[
            dict(
                type='line',
                yref='paper', y0=0, y1=1,
                xref='x', x0=3, x1=3,
                line=dict(color="Black", width=0.5, dash="dash"),
            )
        ])

        # Add sum values
        pop_sum_before_3km = (
            df2[df2['ring'].isin(['1 km', '2 km', '3 km'])]['pop_per_ring'].sum() -
            df1[df1['ring'].isin(['1 km', '2 km', '3 km'])]['pop_per_ring'].sum()
        )
        pop_sum_after_3km = (
            df2[~df2['ring'].isin(['1 km', '2 km', '3 km'])]['pop_per_ring'].sum() -
            df1[~df1['ring'].isin(['1 km', '2 km', '3 km'])]['pop_per_ring'].sum()
        )

        def format_signed(value):
            sign = "+" if value >= 0 else "-"
            return f"{sign}{abs(round(value,-1)):,.0f}".replace(",", "\u00A0")

        fig.add_annotation(
            x=1.95,  # slightly to the left of x=3
            y=0.03,     # bottom of plot
            xref='x',
            yref='paper',
            text=format_signed(pop_sum_before_3km),
            showarrow=False,
            font=dict(color='black')
        )

        fig.add_annotation(
            x=5.05,  # slightly to the right of x=3
            y=0.03,
            xref='x',
            yref='paper',
            text=format_signed(pop_sum_after_3km),
            showarrow=False,
            font=dict(color='black')
        )

        return fig
    fig = plot_muutos(den0,den1)
    return st.plotly_chart(fig, use_container_width=True)



#with den_holder:
#    generate_den_graphs(den0,den1)
    
st.caption("data: [stat.fi](https://www.stat.fi/org/avoindata/paikkatietoaineistot/tilastoruudukko_1km.html)")


#footer
st.markdown('---')
footer_title = '''
[![MIT license](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/teemuja/)
'''
st.markdown(footer_title)
