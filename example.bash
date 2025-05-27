echo "decisionflow for mta using gpt4o"
python decisionflow_main.py \
  --dataset mta \
  --model closed_source \
  --model_path gpt-4o-2024-08-06 \
  --mta_method decisionflow \
  --mta_alignment high

echo "evaluate results for mta"
python decisionflow_main.py \
  --action evaluation \
  --dataset mta \
  --mta_method decisionflow \
  --mta_alignment high

echo "decisionflow for agriculture using gpt4o"
python decisionflow_main.py \
  --dataset agriculture \
  --model closed_source \
  --model_path gpt-4o-2024-08-06 \
  --dellma_mode decisionflow

echo "evaluate results for agriculture"
python decisionflow_main.py \
  --action evaluation \
  --dataset agriculture \
  --dellma_mode decisionflow

echo "decisionflow for stocks using gpt4o"
python decisionflow_main.py \
  --dataset stocks \
  --model closed_source \
  --model_path gpt-4o-2024-08-06 \
  --dellma_mode decisionflow

echo "evaluate results for stocks"
python decisionflow_main.py \
  --action evaluation \
  --dataset stocks \
  --dellma_mode decisionflow