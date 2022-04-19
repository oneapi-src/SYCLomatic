#!/bin/bash

test_source_dir=$1
test_dest_dir=$2
source_files_dir=$test_source_dir/../../../lib/DPCT

#From HelperFeatureEnum.inc, extract all feature enum names to all_feature_enums.txt
#These enum definitions are between line "#ifdef DPCT_FEATURE_ENUM" and line "#endif // DPCT_FEATURE_ENUM"
#Finally remove the comma at the end of each line
awk '/#ifdef DPCT_FEATURE_ENUM\s*$/{flag=1; next} /#endif \/\/ DPCT_FEATURE_ENUM\s*$/{flag=0}  {if(flag==1){print $0}}' $test_dest_dir/../../../../include/clang/DPCT/HelperFeatureEnum.inc | awk -F ',' '{print $1}' > $test_dest_dir/all_feature_enums.txt

#Grep each feature enum name in source code folder
#If something is greped, save the feature enum name in feature_used_in_src_code_temp.txt
touch $test_dest_dir/feature_used_in_src_code_temp.txt
temp_lines=$(cat $test_dest_dir/all_feature_enums.txt)
for feature in $temp_lines
do
  if grep -qrnw $feature $source_files_dir
  then
    echo $feature >> $test_dest_dir/feature_used_in_src_code_temp.txt
  fi
done

#Sort and uniq all the feature used in source code and save the result in feature_used_in_src_code.txt
cat $test_dest_dir/feature_used_in_src_code_temp.txt | sort | uniq > $test_dest_dir/feature_used_in_src_code.txt

#Grep the keyword "TEST_FEATURE" in test folder to get all features which are tested and save the result in feature_in_test.txt
grep -r "TEST_FEATURE" --include api_test*.cu $test_source_dir | awk '{print $NF}' | sort | uniq > $test_dest_dir/feature_in_test.txt

#Compare above two files. The diff result should be empty.
echo "begin" > $test_dest_dir/result.txt
diff --new-line-format="" --unchanged-line-format=""  $test_dest_dir/feature_used_in_src_code.txt  $test_dest_dir/feature_in_test.txt >> $test_dest_dir/result.txt
echo "end" >> $test_dest_dir/result.txt

#Remove the temp file
rm $test_dest_dir/feature_used_in_src_code_temp.txt
