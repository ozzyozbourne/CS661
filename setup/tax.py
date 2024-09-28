def cal_tax():
    "This functional calculate the federal, state tax"
    fed_tax = float(input("Enter the federal tax please: "))
    state_tax = float(input("Enter the state tax please: "))
    salary = float(input("Enter your salary please: "))

    total_tax = (salary * fed_tax/ 100) + (salary * state_tax / 100)
    salary = salary - total_tax

    print(f'Your total tax is {total_tax} and your salary after tax is {salary}')