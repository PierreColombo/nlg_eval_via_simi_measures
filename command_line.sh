for metric in baryscore depthscore infolm; do #depthscore baryscore
  for measure_to_use in kl alpha ab; do
    python score_cli.py --ref="samples/refs.txt" --cand="samples/hyps.txt" --metric_name=${metric} --measure_to_use=${measure_to_use}
    python score_cli.py --ref="samples/refs.txt" --cand="samples/hyps.txt" --metric_name=${metric} --idf --measure_to_use=${measure_to_use}
  done
done
