"""팀원별 가드 플러그인 (member plugins).

member_a ~ member_d 폴더 각각이 하나의 플러그인 패키지다. 이 패키지 아래의 하위 패키지는
registry 가 서버 시작 시 자동으로 import 하므로, 각자 guards.py 에 @register_guard 클래스를 추가하면 된다.
가드 이름은 "member_x.<이름>" 형식으로 지어 다른 팀원과 겹치지 않게 한다.
"""
