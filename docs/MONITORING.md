# Render 서비스 모니터링 및 슬립 방지 가이드

Render의 무료 플랜은 일정 시간 동안 트래픽이 없으면 자동으로 서비스가 슬립 상태가 됩니다. 이 문서는 서비스를 지속적으로 활성 상태로 유지하는 방법을 설명합니다.

## 내장된 자동 핑(Ping) 기능

이 프로젝트에는 이미 자동 핑 기능이 구현되어 있습니다:

1. `/ping` 엔드포인트: 서버 상태를 확인할 수 있는 엔드포인트입니다.
2. 자동 핑 태스크: 서버가 시작될 때 백그라운드에서 실행되며, 5분마다 자체적으로 핑을 수행합니다.

이 기능은 기본적인 슬립 방지 기능을 제공하지만, 서버 자체에서 실행되므로 서버가 이미 슬립 상태인 경우 효과가 없을 수 있습니다.

## 외부 모니터링 서비스 사용하기

보다 안정적인 모니터링을 위해 다음과 같은 무료 외부 서비스를 사용할 수 있습니다:

### UptimeRobot 설정 방법

1. [UptimeRobot](https://uptimerobot.com/)에 가입합니다.
2. '+ Add New Monitor' 버튼을 클릭합니다.
3. 모니터 유형으로 'HTTP(s)'를 선택합니다.
4. 다음 정보를 입력합니다:
   - 이름: 'Construction Chatbot API'
   - URL: `https://construction-chatbot-api.onrender.com/ping`
   - 모니터링 간격: 5분 (무료 플랜 기준)
5. 'Create Monitor'를 클릭하여 모니터링을 시작합니다.

### Pingdom 설정 방법

1. [Pingdom](https://www.pingdom.com/)에 가입합니다.
2. '+ Add New Check' 버튼을 클릭합니다.
3. 'Web check'를 선택합니다.
4. 다음 정보를 입력합니다:
   - 이름: 'Construction Chatbot API'
   - URL: `https://construction-chatbot-api.onrender.com/ping`
   - 테스트 빈도: 1분 (유료 플랜)
5. 'Add Check'를 클릭하여 모니터링을 시작합니다.

## Render 유료 플랜으로 업그레이드

궁극적으로, 프로덕션 환경에서는 Render의 유료 플랜으로 업그레이드하는 것이 가장 안정적인 해결책입니다. 유료 플랜은 슬립 상태로 전환되지 않고 항상 활성 상태를 유지합니다.

### 유료 플랜 장점:
- 지속적인 가용성
- 더 빠른 응답 시간
- 첫 요청에 대한 Cold Start 지연 없음
- 더 많은 CPU, RAM 및 대역폭

## 고급: GitHub Actions를 사용한 모니터링

GitHub Actions를 사용하여 주기적으로, 예를 들어 10분마다 서비스를 핑하는 워크플로우를 설정할 수도 있습니다. 이 방법은 무료이며 GitHub 계정이 있는 경우 설정할 수 있습니다.

1. 프로젝트의 `.github/workflows/` 디렉토리를 생성합니다.
2. 다음 내용으로 `ping.yml` 파일을 생성합니다:

```yaml
name: Keep Render Service Alive

on:
  schedule:
    # 10분마다 실행
    - cron: '*/10 * * * *'

jobs:
  ping:
    runs-on: ubuntu-latest
    steps:
      - name: Ping Render service
        run: curl -s https://construction-chatbot-api.onrender.com/ping
      - name: Print time
        run: echo "Pinged at $(date)"
``` 