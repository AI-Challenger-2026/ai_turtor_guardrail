---
doc_id: py101-u4
title: 파이썬 입문
unit: 4단원. 함수
---

# 4단원. 함수

## 4.1 함수 정의와 호출

함수(function)는 특정 작업을 하는 코드 묶음에 이름을 붙여 둔 것이다. 같은 코드를 여러 번 쓰지 않고 필요할 때마다 이름으로 불러 쓸 수 있어 코드가 짧아지고 고치기 쉬워진다. `print()`, `len()`처럼 파이썬이 미리 만들어 둔 함수를 내장 함수라고 하고, 직접 만든 함수를 사용자 정의 함수라고 한다.

함수는 `def` 키워드로 정의한다. `def 함수이름(매개변수):` 다음 줄부터 들여쓰기한 코드가 함수의 본문이다. 정의만 해서는 실행되지 않고, `함수이름(인자)`처럼 호출해야 실행된다.

```python
def greet(name):
    print(f"안녕, {name}!")

greet("민지")   # 안녕, 민지!
greet("준호")   # 안녕, 준호!
```

### 매개변수와 인자

매개변수(parameter)는 함수를 정의할 때 괄호 안에 적는 변수이고, 인자(argument)는 함수를 호출할 때 실제로 넘겨주는 값이다. 위 예에서 `name`은 매개변수, `"민지"`는 인자다. 매개변수는 쉼표로 여러 개 둘 수 있으며, 호출할 때는 매개변수 개수에 맞게 인자를 넘겨야 한다. 개수가 맞지 않으면 `TypeError`가 발생한다.

## 4.2 return과 반환값

`return`은 함수의 실행 결과를 함수를 호출한 곳으로 돌려주는 문장이다. 돌려주는 값을 반환값(return value)이라고 한다. `return`을 만나면 함수는 그 자리에서 즉시 끝난다.

```python
def add(a, b):
    return a + b

result = add(3, 4)
print(result * 2)   # 14
```

`return`이 없는 함수를 호출하면 `None`이 반환된다.

### print와 return의 차이

`print()`는 값을 화면에 보여 줄 뿐이고, 그 값을 다른 계산에 다시 쓸 수 없다. 반면 `return`은 값을 돌려주므로 변수에 저장하거나 다른 계산에 이어서 쓸 수 있다.

```python
def square_print(x):
    print(x * x)

def square_return(x):
    return x * x

a = square_print(3)    # 화면에 9 출력, a 는 None
b = square_return(3)   # 화면 출력 없음, b 는 9
```

함수가 여러 값을 돌려줘야 할 때는 `return 몫, 나머지`처럼 쉼표로 나열한다. 이때 값들은 튜플 하나로 묶여 반환되며, `q, r = divide(7, 2)`처럼 언패킹해서 받을 수 있다.

## 4.3 기본값 매개변수와 변수의 범위

매개변수에 기본값(default value)을 정해 두면, 호출할 때 그 인자를 생략할 수 있다. 기본값이 있는 매개변수는 기본값이 없는 매개변수보다 뒤에 와야 한다.

```python
def power(base, exp=2):
    return base ** exp

print(power(3))      # 9  (exp 는 기본값 2)
print(power(2, 5))   # 32
print(power(exp=3, base=2))  # 8  (키워드 인자는 순서를 바꿔도 된다)
```

`power(exp=3, base=2)`처럼 `매개변수이름=값` 형태로 넘기는 인자를 키워드 인자라고 한다.

### 지역 변수와 전역 변수

함수 안에서 만든 변수는 지역 변수(local variable)로, 그 함수 안에서만 쓸 수 있고 함수가 끝나면 사라진다. 함수 밖에서 만든 변수는 전역 변수(global variable)로, 함수 안에서 값을 읽을 수는 있다. 하지만 함수 안에서 전역 변수에 새 값을 대입하면 같은 이름의 지역 변수가 새로 만들어질 뿐 전역 변수는 바뀌지 않는다.

```python
count = 0

def increase():
    count = 10      # 새로운 지역 변수 count 가 만들어진다
    print(count)    # 10

increase()
print(count)        # 0 (전역 변수는 그대로)
```

함수 안에서 전역 변수를 꼭 바꿔야 하면 `global count`처럼 선언할 수 있지만, 코드의 흐름을 추적하기 어려워지므로 되도록 매개변수와 `return`으로 값을 주고받는 방식을 권장한다.
