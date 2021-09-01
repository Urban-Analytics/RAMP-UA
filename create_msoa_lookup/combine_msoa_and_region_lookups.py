import pandas as pd


def main():
    area_name = 'London'
    # MSOAS_df = get_MSOAs_by_region(area_name)
    MSOAS_df = get_MSOAS_by_ITL(area_name)

    # get cases for each region
    seed_day_index = 20
    regions_with_cases_df = merge_with_cases(MSOAS_df, seed_day_index)

    # check for duplicates
    deduplicated_df = regions_with_cases_df.drop_duplicates(subset=['MSOA11CD'])
    print(f"deduped length: {len(deduplicated_df.index)} MSOAs")
    assert len(deduplicated_df.index) == len(regions_with_cases_df.index)

    # write MSOAs with cases to csv file
    output_name = "./" + area_name.replace(" ", "_") + "_with_cases.csv"
    regions_with_cases_df.to_csv(output_name, index=False)


def get_MSOAS_by_ITL(ITL_name):
    print("\nloading complete lookup")

    complete_lookup = pd.read_csv('../data/raw_data/reference_data/lookUp.csv')
    ITL1_df = complete_lookup.drop(['MSOA11CD', 'MSOA11NM', 'LAD20CD', 'LAD20NM', 'ITL321CD', 'ITL321NM',
       'ITL221CD', 'ITL221NM', 'CTY20NM', 'CCTY20NM',
       'GoogleMob', 'OSM', 'OldTU', 'NewTU'], axis=1)\
        .drop_duplicates(subset=['ITL121CD'])

    print(ITL1_df.head())
    print(ITL1_df.columns)
    print(len(ITL1_df.index))

    # group msoas by ITL1
    ITL_groups = complete_lookup.groupby(['ITL121NM'])
    print(ITL_groups.groups.keys())
    ITL_grouped = ITL_groups.get_group(ITL_name)
    msoas_for_ITL = ITL_grouped['MSOA11CD']

    return msoas_for_ITL


def get_MSOAs_by_region(region_name):
    msoa_to_lad_lookup = load_msoa_to_lad_lookup()
    lad_to_region_lookup = load_lad_to_region_lookup()

    print("\nMerging on LAD")
    merged_lookup_df = pd.merge(
        msoa_to_lad_lookup,
        lad_to_region_lookup,
        how="inner",
        on='LAD11CD'
    )

    # group msoas by region
    region_groups = merged_lookup_df.groupby(['RGN11NM'])
    print(region_groups.groups.keys())
    region_grouped = region_groups.get_group(region_name)
    msoas_for_region = region_grouped['MSOA11CD']

    return msoas_for_region

def load_msoa_to_lad_lookup():
    print("\nimporting msoa to lad lookup")
    msoa_to_lad_lookup = pd.read_csv('raw_data/MSOA_to_LAD_lookup.csv')

    msoa_to_lad_lookup = msoa_to_lad_lookup.drop(['OA11CD', 'LSOA11CD', 'LSOA11NM', 'LAD11NMW', 'ObjectId'], axis=1)\
        .drop_duplicates(subset=['MSOA11CD'])

    # print(msoa_to_lad_lookup.head())
    # print(msoa_to_lad_lookup.columns)
    # print(len(msoa_to_lad_lookup.index))

    return msoa_to_lad_lookup


def load_lad_to_region_lookup():
    print("\nimporting lad to region lookup")
    lad_to_region_lookup = pd.read_csv('raw_data/LAD_to_region_lookup_2011.csv')

    lad_to_region_lookup = lad_to_region_lookup.drop(['OA11CD', 'BUASD11CD', 'BUASD11NM', 'BUA11CD', 'BUA11NM', 'LAD11NMW', 'RGN11NMW', 'ObjectId'], axis=1)\
        .drop_duplicates(subset=['LAD11CD'])

    # print(lad_to_region_lookup.head())
    # print(lad_to_region_lookup.columns)
    # print(len(lad_to_region_lookup.index))

    return lad_to_region_lookup


def merge_with_cases(msoas_df, seed_day_index=20):
    initial_cases_df = pd.read_csv('raw_data/england_initial_cases_MSOAs.csv')
    # print(initial_cases_df.head())

    msoas_with_cases_df = pd.merge(
        msoas_df,
        initial_cases_df,
        how="inner",
        on="MSOA11CD"
    )

    # print(merged_df.head())
    # print(len(merged_df.index))

    case_column_name = f"D{seed_day_index}"

    msoas_with_cases_df = msoas_with_cases_df[["MSOA11CD", case_column_name]]
    msoas_with_cases_df = msoas_with_cases_df.rename(columns={case_column_name: "cases"})
    print(msoas_with_cases_df.head())
    print(len(msoas_with_cases_df.index))

    return msoas_with_cases_df


if __name__ == "__main__":
    main()
