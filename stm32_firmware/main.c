/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : STM32F4VE - ESP-01 UART to LED D2 (PA6)
  *                   USART2: PA2 (TX), PA3 (RX) - Baud: 115200
  *                   LED D2: PA6 (Active LOW: RESET = ON, SET = OFF)
  ******************************************************************************
  */
/* USER CODE END Header */

#include "main.h"
#include <string.h>
#include <stdio.h>

UART_HandleTypeDef huart2;

#define RX_BUF_SIZE      128
#define LED_ON_DURATION  10000 /* Sáng 10 giây */

uint8_t          rx_byte;
char             rx_buf[RX_BUF_SIZE];
uint16_t         rx_idx     = 0;
volatile uint8_t line_ready = 0;

void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);
void LED_On_Alert(uint32_t duration_ms);
void Process_UART_Line(char *line);

int main(void)
{
  HAL_Init();
  SystemClock_Config();
  MX_GPIO_Init();
  MX_USART2_UART_Init();

  /* 1. KHỞI ĐỘNG: Tắt LED D2 */
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, GPIO_PIN_SET); /* SET = TẮT */

  /* 2. TEST PHẦN CỨNG: Nháy nhanh 3 lần để báo STM32 đã nạp code & đang chạy */
  for (int i = 0; i < 3; i++) {
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, GPIO_PIN_RESET); /* SÁNG */
    HAL_Delay(100);
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, GPIO_PIN_SET);   /* TẮT */
    HAL_Delay(100);
  }

  /* 3. BẮT ĐẦU CHỜ NHẬN DỮ LIỆU TỪ ESP-01 */
  HAL_UART_Receive_IT(&huart2, &rx_byte, 1);

  while (1)
  {
    /* Đảm bảo ngắt luôn mở */
    if (huart2.RxState != HAL_UART_STATE_BUSY_RX)
    {
      __HAL_UART_CLEAR_OREFLAG(&huart2);
      __HAL_UART_CLEAR_NEFLAG(&huart2);
      __HAL_UART_CLEAR_FEFLAG(&huart2);
      __HAL_UART_CLEAR_PEFLAG(&huart2);
      HAL_UART_Receive_IT(&huart2, &rx_byte, 1);
    }

    if (line_ready)
    {
      line_ready = 0;
      Process_UART_Line(rx_buf);
      rx_idx = 0;
      memset(rx_buf, 0, sizeof(rx_buf));
      HAL_UART_Receive_IT(&huart2, &rx_byte, 1);
    }
  }
}

void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState       = RCC_HSE_ON;
  RCC_OscInitStruct.PLL.PLLState   = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource  = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM       = 8;
  RCC_OscInitStruct.PLL.PLLN       = 336;
  RCC_OscInitStruct.PLL.PLLP       = RCC_PLLP_DIV2;
  RCC_OscInitStruct.PLL.PLLQ       = 4;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  RCC_ClkInitStruct.ClockType      = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                                    |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource   = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider  = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV4;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_5) != HAL_OK)
  {
    Error_Handler();
  }
}

static void MX_USART2_UART_Init(void)
{
  huart2.Instance          = USART2;
  huart2.Init.BaudRate     = 115200;
  huart2.Init.WordLength   = UART_WORDLENGTH_8B;
  huart2.Init.StopBits     = UART_STOPBITS_1;
  huart2.Init.Parity       = UART_PARITY_NONE;
  huart2.Init.Mode         = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl    = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
}

static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  __HAL_RCC_GPIOA_CLK_ENABLE();

  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, GPIO_PIN_SET); /* Mặc định TẮT */

  GPIO_InitStruct.Pin   = GPIO_PIN_6;
  GPIO_InitStruct.Mode  = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull  = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);
}

void Process_UART_Line(char *line)
{
  LED_On_Alert(LED_ON_DURATION);
}

void LED_On_Alert(uint32_t duration_ms)
{
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, GPIO_PIN_RESET); /* RESET = SÁNG */
  HAL_Delay(duration_ms);
  HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, GPIO_PIN_SET);   /* SET = TẮT */
}

/* Callback khi nhận được byte UART thành công */
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART2)
  {
    /* BẬT SÁNG LED NGAY KHI CÓ BẤT KỲ TÍN HIỆU NÀO ĐẾN */
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, GPIO_PIN_RESET);

    if (rx_idx < RX_BUF_SIZE - 1)
    {
      rx_buf[rx_idx++] = rx_byte;
      rx_buf[rx_idx]   = '\0';
    }

    line_ready = 1;
    HAL_UART_Receive_IT(&huart2, &rx_byte, 1);
  }
}

/* Callback khi có lỗi tín hiệu (như quẹt dây vào GND tạo xung nhiễu) */
void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
  if (huart->Instance == USART2)
  {
    /* BẬT SÁNG LED NGAY KHI PHÁT HIỆN XUNG TÍN HIỆU / QUẸT GND */
    HAL_GPIO_WritePin(GPIOA, GPIO_PIN_6, GPIO_PIN_RESET);

    __HAL_UART_CLEAR_OREFLAG(huart);
    __HAL_UART_CLEAR_NEFLAG(huart);
    __HAL_UART_CLEAR_FEFLAG(huart);
    __HAL_UART_CLEAR_PEFLAG(huart);
    HAL_UART_Receive_IT(huart, &rx_byte, 1);
  }
}

void Error_Handler(void)
{
  __disable_irq();
  while (1) {}
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line) {}
#endif