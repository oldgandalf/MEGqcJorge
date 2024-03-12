# import questionary

# # Define the categories and subcategories
# categories = {
#     'sub': {'029', '026', '025', '021', '016', '028', '022', '030', '018', '020', '009', '015', '013', '027', '014', '017', '012', '019', '023', '024', '031'},
#     'ses': {'1'},
#     'task': {'induction', 'deduction'},
#     'run': {1},
#     'desc': {'REPORT', 'ECGs', 'EOGs', 'Muscle', 'epochs', 'Sensors', 'PSDs', 'STDs', 'SimpleMetrics'}
# }

# # Create a list of subcategories with category titles
# subcategories = []
# for category, items in categories.items():
#     subcategories.append(questionary.Separator(f'== {category} =='))
#     for item in items:
#         subcategories.append(str(item))

# # Ask the user to select a subcategory
# subcategory = questionary.select("Select a subcategory:", choices=subcategories).ask()

# print('You selected:', subcategory)



from prompt_toolkit.shortcuts import checkboxlist_dialog
from prompt_toolkit.styles import Style

# Define the categories and subcategories
categories = {
    'METRIC': ['_ALL_metrics_', 'ECGs', 'EOGs', 'Muscle', 'PSDs', 'STDs'],
    'SUBJECT': ['_ALL_subjects_', '029', '026', '025', '021', '016', '028', '022', '030', '018', '020', '009', '015', '013', '027', '014', '017', '012', '019', '023', '024', '031'],
    'SESSION': ['_ALL_sessions_', '1'],
    'TASK': ['_ALL_tasks_', 'induction', 'deduction'],
    'RUN': ['_ALL_runs_', 1]
}

# Create a list of values with category titles
values = []
for category, items in categories.items():
    values.append((f'== {category} ==', f'== {category} =='))
    for item in items:
        values.append((str(item), str(item)))

results = checkboxlist_dialog(
    title="CheckboxList dialog",
    text="Select subcategories:",
    values=values,
    style=Style.from_dict({
        'dialog': 'bg:#cdbbb3',
        'button': 'bg:#bf99a4',
        'checkbox': '#e8612c',
        'dialog.body': 'bg:#a9cfd0',
        'dialog shadow': 'bg:#c98982',
        'frame.label': '#fcaca3',
        'dialog.body label': '#fd8bb6',
    })
).run()

# Ignore the category titles
selected_subcategories = [result for result in results if not result.startswith('== ')]

print('You selected:', selected_subcategories)