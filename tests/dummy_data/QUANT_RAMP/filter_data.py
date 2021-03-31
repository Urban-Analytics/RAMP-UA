# Used to filter out areas in the QUANT data to create dummy data for testing.
# Reads one of the QUANT input files and looks for lines that represent a particular area (i.e. areas that
# are used in the RAMP test data). It writes any matching lines to the console.


# TWO IMPORTANT THINGS TO NOTE!
#
# 1. It will write out lines like follows, e.g. for area 4:
#
# 4,E02004300,490.0,4226,54.875492,-1.7496299,416064.06,553378.94,11946803.0
#
# The 4th item in the list (4226) is the zoneID. That needs to be changed manually to reflect the ID at the
# start of the line. I.e. that line needs to be rewritten to:
#
# 4,E02004300,490.0,4,54.875492,-1.7496299,416064.06,553378.94,11946803.0
#
# 2. The zone IDs need to count up contiquously from 0. So in that previous example, zone 4 actually needed changing
# to zone 3 as in the retail example there is no zone 2. In the school examples the zones are around 15, so all need
# changing to run from 0 to 1.




input_file = "primaryPopulation.csv"
#input_file = "secondaryPopulation.csv"
#input_file = "retailpointsPopulation.csv"

AREAS = ['E02004297', 'E02004290', 'E02004298', 'E02004299', 'E02004291', 'E02004300', 'E02004292', 'E02004301']

print(f"Areas: {AREAS}")
matched_lines = 0

with open(input_file, 'r') as f:
    for line in f:
        if any(area in line for area in AREAS):
            print(line.strip())
            matched_lines += 1

assert matched_lines == len(AREAS), f"Warning, there are {len(AREAS)} to look for, but only {matched_lines} lines found"
print("Finished")
