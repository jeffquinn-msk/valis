import pandas
import geopandas as gpd
from shapely import Polygon
from nuc2seg.utils import (
    drop_invalid_geometries,
    filter_gdf_to_intersects_polygon,
    filter_gdf_to_inside_polygon,
)
import anndata
import os

def main():
    data_dir = "/Users/quinnj2/Documents/epithelioid_sarcoma/data/551"
    dst_dir = f"/Users/quinnj2/Documents/epithelioid_sarcoma/runs/valis_2025-10-13"
    boundaries = pandas.read_csv(os.path.join(dst_dir, "TMA2_ES2_polygons_warped.csv"))
    metadata = pandas.read_csv(os.path.join(dst_dir, "TMA2_ES2_metadata_file.csv.gz"))

    geo_df = gpd.GeoDataFrame(
            boundaries,
            geometry=gpd.points_from_xy(
                boundaries['x_global_px'], boundaries['y_global_px']
            ),
        )

    geo_df = geo_df.groupby('cell').filter(lambda x: len(x) > 2)
    cell_id_column = "cell"


    polys = geo_df.groupby(cell_id_column).agg({
            "geometry": lambda x: Polygon(x.tolist()),
            "cellID": max,
        "fov": max}
        )
    gdf = drop_invalid_geometries(gpd.GeoDataFrame(polys.reset_index(drop=True)))

    colnames = ['4-1BB',
     'B7-H3',
     'Bcl-2',
     'Beta-catenin',
     'CCR7',
     'CD11b',
     'CD11c',
     'CD123',
     'CD127',
     'CD138',
     'CD14',
     'CD15',
     'CD16',
     'CD163',
     'CD19',
     'CD20',
     'CD27',
     'CD3',
     'CD31',
     'CD34',
     'CD38',
     'CD39',
     'CD4',
     'CD40',
     'CD45',
     'CD45RA',
     'CD56',
     'CD68',
     'CD8',
     'CTLA4',
     'Channel-CD45',
     'Channel-DNA',
     'Channel-G',
     'Channel-Membrane',
     'Channel-PanCK',
     'EGFR',
     'EpCAM',
     'FABP4',
     'FOXP3',
     'Fibronectin',
     'GITR',
     'GZMA',
     'GZMB',
     'HLA-DR',
     'Her2',
     'ICAM1',
     'ICOS',
     'IDO1',
     'IL-18',
     'IL-1b',
     'IgD',
     'Ki-67',
     'LAG3',
     'LAMP1',
     'NF-kB p65',
     'PD-1',
     'PD-L1',
     'PD-L2',
     'SMA',
     'STING',
     'TCF7',
     'Tim-3',
     'VISTA',
     'Vimentin',
     'iNOS',
     'p53',
     'pan-RAS',
     'Ms IgG1',
     'Rb IgG']

    exprmat = pandas.read_csv(os.path.join(data_dir, "TMA2_ES2_exprMat_file.csv.gz"))
    adata = anndata.AnnData(X=exprmat[colnames].values)
    adata.var_names = colnames

    adata.obs_names = exprmat.index
    adata.obs = exprmat[["fov","cell_ID"]]

    x_coord = adata.obs.merge(gdf, left_on=["cell_ID","fov"], right_on=["cellID","fov"], how="left").geometry.apply(lambda x: x.centroid.x if x else pandas.NA)
    y_coord = adata.obs.merge(gdf, left_on=["cell_ID","fov"], right_on=["cellID","fov"], how="left").geometry.apply(lambda x: x.centroid.y if x else pandas.NA)

    adata.obs['x_centroid'] = x_coord.values
    adata.obs['y_centroid'] = y_coord.values
    adata = adata[adata.obs.x_centroid.notna(),:].copy()
    adata.obs['x_centroid'] = adata.obs['x_centroid'].astype(float)
    adata.obs['y_centroid'] = adata.obs['y_centroid'].astype(float)
    adata.obsm['spatial'] = adata.obs[['x_centroid','y_centroid']].values
    adata.obs.join(metadata,left_on=["cell_ID","fov"], right_on=["cell_ID","fov"], how="left")
    adata.write_h5ad(os.path.join(dst_dir, "COSMX_segmentations.h5ad"))
    gdf.to_parquet(os.path.join(dst_dir, "COSMX_segmentations.parquet"))

if __name__ == "__main__":
    main()