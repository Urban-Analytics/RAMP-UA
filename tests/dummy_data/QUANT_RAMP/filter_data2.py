# After running filter_data2 to find out which areas are included in the test data,
# you will now also be able to see which zone numbers these areas refer to (the zonei column)
# So now you can run this code to get rid of those zones from the *Zones.csv files.


input_file = "primaryZones.csv"
#INPUT_ZONES = [15, 16, 17, 22, 24, 25, 26]

#input_file = "secondaryZones.csv"
#INPUT_ZONES = [15, 16, 17, 22, 24, 25, 26]

#input_file = "retailpointsZones.csv"
INPUT_ZONES = [0, 1, 2, 3, 4, 5, 6, 7]

print(f"Zones: {INPUT_ZONES}")
matched_lines = 0

with open(input_file, 'r') as f:
    for lineno, line in enumerate(f):
        if lineno == 0: # Skip header line
            continue
        line_array = line.split(',')
        # Need to check that the origin (first item in the list) *and* destination (third iterm)
        # are both in the INPUT_ZONES
        #if any(zone == int(line_array[0]) for zone in INPUT_ZONES) and any(zone == int(line_array[2]) for zone in INPUT_ZONES):

        if any(zone == int(line_array[0]) for zone in INPUT_ZONES) :
            print(line.strip())
            matched_lines += 1

assert matched_lines == len(INPUT_ZONES), f"Warning, there are {len(INPUT_ZONES)} to look for, but only {matched_lines} lines found"
print("Finished")
