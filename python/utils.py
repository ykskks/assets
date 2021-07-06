def update_tracking(
    model_id, field, value, csv_file="logs/history.csv", integer=False, digits=None
):
    try:
        df = pd.read_csv(csv_file, index_col=[0])
    except:
        df = pd.DataFrame()

    if integer:
        value = round(value)
    elif digits is not None:
        value = round(value, digits)
    df.loc[model_id, field] = value  # Model number is index
    df.to_csv(csv_file)
