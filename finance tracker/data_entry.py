from datetime import datetime

#specify the date format
data_format = "%d-%m-%Y"
CATEGORY_OPTIONS = {"I": "Income", "E": "Expense"}

#get the date from the user
def get_date(prompt, allow_default=False):
    try:
        date_str = input(prompt)
        if allow_default and not date_str:
            return datetime.today().strftime(data_format)
        else:
            return date_str
    except ValueError:
        print("Invalid date format. Please use DD-MM-YYYY")
        return get_date(prompt, allow_default)

#get the amount from the user
def get_amount():
    try:
        amount = float(input("Enter the amount: "))
        if amount < 0:
            print("Amount cannot be negative. Please enter a positive amount.")
            return get_amount()
        return amount
    except ValueError:
        print("Invalid amount format. Please use numbers")
        return get_amount()

#get the category from the user
def get_category():
    category = input("Enter the category ('I' for Income or 'E' for expense): ").upper()
    if category not in ['I', 'E']:
        print("Invalid category. Please enter 'I' for Income or 'E' for expense.")
        return get_category()
    if category in CATEGORY_OPTIONS:
        return CATEGORY_OPTIONS[category]
    else:
        print("Invalid category. Please enter 'I' for Income or 'E' for expense.")
        return get_category()

#get the description from the user
def get_description():
    return input("Enter the description (optional): ")
