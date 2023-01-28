import cv2
import mediapipe as mp
import os
import time

class PatternRecognizer:
    def __init__(self):
        self._mp_drawing = mp.solutions.drawing_utils
        self._mp_drawing_styles = mp.solutions.drawing_styles
        self._mp_hands = mp.solutions.hands
        self._selected_nodes = []
        self._selected_nodes_id = []

        self._clear_btn_color = (0, 0, 225)
        self._video_width, self._video_height = 1200, 800
        
        
        self._check_image = cv2.imread(os.path.join(os.getcwd(), "images", "check.png"))
        self._check_image = cv2.resize(self._check_image, (self._video_width, self._video_height))
        # self._check_image = cv2.flip(self._check_image, 1)
        # y = self._check_image.shape[1]
        # x = self._check_image.shape[0]
        # r = 10
        # self._check_image = self._check_image[y:(y+2*r), x:(x+2*r)]
        
        self._last_index_finger_coord = None

        
        self._gt_path = [f for f in [2, 4, 5, 1]] #start index from 0
    
    def _make_gt_visible(self, image, nodes):
        for index, nod_num in enumerate(self._gt_path):
            try:
                cv2.line(image, nodes[nod_num][0], nodes[self._gt_path[index+1]][0], (0, 225, 0), 2)    
            except:
                continue
    
    def _read_stream(self, id_):
        cap = cv2.VideoCapture(id_)

        return cap

    def _calc_nodes(self, frame_w, frame_h):
        center_x, center_y = frame_w // 2, frame_h //2
        offset_x, offset_y = 50, 50

        
        
        # node_0 = (int(center_y - offset_y * 1), int(center_x - offset_x * 1))

        node_0 =  ((int(frame_h * 0.5), int(frame_w * 0.1)), 0)
        node_1 =  ((int(frame_h * 0.75), int(frame_w * 0.1)), 1)
        node_2 =  ((int(frame_h * 1.0), int(frame_w * 0.1)), 2)

        node_3 =  ((int(frame_h * 0.5), int(frame_w * 0.3)), 3)
        node_4 =  ((int(frame_h * 0.75), int(frame_w * 0.3)), 4)
        node_5 =  ((int(frame_h * 1.0), int(frame_w * 0.3)), 5)

        node_6 =  ((int(frame_h * 0.5), int(frame_w * 0.5)), 6)
        node_7 =  ((int(frame_h * 0.75), int(frame_w * 0.5)), 7)
        node_8 =  ((int(frame_h * 1.0), int(frame_w * 0.5)), 8)
        
        nodes = [node_0, node_1, node_2, node_3, node_4, node_5, node_6, node_7, node_8, node_8]

        return nodes
            
    def _draw_nodes(self, image, nodes):
        for node in nodes:
            cv2.circle(image, node[0], 20, (233, 129, 55), -1)


    def _draw_landmarks(self, image, hand_landmarks):
        
        self._mp_drawing.draw_landmarks(
            image,
            hand_landmarks,

            self._mp_hands.HAND_CONNECTIONS,
            self._mp_drawing_styles.get_default_hand_landmarks_style(),
            self._mp_drawing_styles.get_default_hand_connections_style()
            )

    def _set_path(self, nodes, x_index_finger, y_index_finger):
        
        node_0, node_1, node_2, node_3, node_4, node_5, node_6, node_7, node_8 = nodes[0], nodes[1], nodes[2], nodes[3], nodes[4], nodes[5], nodes[6], nodes[7], nodes[8]

        if (abs(node_0[0][0]) - abs(x_index_finger)) ** 2 + (abs(node_0[0][1]) - abs(y_index_finger)) ** 2 <= 20 ** 2:
            if not(node_0[1] in self._selected_nodes_id):
                self._selected_nodes.append(node_0)
                self._selected_nodes_id.append(0)

        if (abs(node_1[0][0]) - abs(x_index_finger)) ** 2 + (abs(node_1[0][1]) - abs(y_index_finger)) ** 2 <= 20 ** 2:
            if not(node_1[1] in self._selected_nodes_id):
                self._selected_nodes.append(node_1)
                self._selected_nodes_id.append(1)

        if (abs(node_2[0][0]) - abs(x_index_finger)) ** 2 + (abs(node_2[0][1]) - abs(y_index_finger)) ** 2 <= 20 ** 2:
            if not(node_2[1] in self._selected_nodes_id):
                self._selected_nodes.append(node_2)
                self._selected_nodes_id.append(2)

        if (abs(node_3[0][0]) - abs(x_index_finger)) ** 2 + (abs(node_3[0][1]) - abs(y_index_finger)) ** 2 <= 20 ** 2:
            if not(node_3[1] in self._selected_nodes_id):
                self._selected_nodes.append(node_3)
                self._selected_nodes_id.append(3)

        if (abs(node_4[0][0]) - abs(x_index_finger)) ** 2 + (abs(node_4[0][1]) - abs(y_index_finger)) ** 2 <= 20 ** 2:
            if not(node_4[1] in self._selected_nodes_id):
                self._selected_nodes.append(node_4)
                self._selected_nodes_id.append(4)
                
        if (abs(node_5[0][0]) - abs(x_index_finger)) ** 2 + (abs(node_5[0][1]) - abs(y_index_finger)) ** 2 <= 20 ** 2:
            if not(node_5[1] in self._selected_nodes_id):
                self._selected_nodes.append(node_5)
                self._selected_nodes_id.append(5)


        if (abs(node_6[0][0]) - abs(x_index_finger)) ** 2 + (abs(node_6[0][1]) - abs(y_index_finger)) ** 2 <= 20 ** 2:
            if not(node_6[1] in self._selected_nodes_id):
                self._selected_nodes.append(node_6)
                self._selected_nodes_id.append(6)

        if (abs(node_7[0][0]) - abs(x_index_finger)) ** 2 + (abs(node_7[0][1]) - abs(y_index_finger)) ** 2 <= 20 ** 2:
            if not(node_7[1] in self._selected_nodes_id):
                self._selected_nodes.append(node_7)
                self._selected_nodes_id.append(7)
                
        if (abs(node_8[0][0]) - abs(x_index_finger)) ** 2 + (abs(node_8[0][1]) - abs(y_index_finger)) ** 2 <= 20 ** 2:
            if not(node_8[1] in self._selected_nodes_id):
                self._selected_nodes.append(node_8)
                self._selected_nodes_id.append(8)

    def _draw_clear_btn(self, image, image_width, image_height):
        rec_w, rec_h = 200, 100

        start_point = (int(image_width - 0.2*image_width), int(image_height - 0.2*image_height))
        end_point = (int(image_width - 0.16*image_width) + rec_w, int(image_height - 0.16*image_height) + rec_h)
        cv2.rectangle(image, start_point, end_point, self._clear_btn_color, thickness=-1)

        return start_point, end_point
            
    def _check_clearance(self, x_index_finger, y_index_finger, clear_btn_start, clear_btn_end):
        # print("x_index_finger", x_index_finger)
        # print("y_index_finger", y_index_finger)
        # print("clear_btn_start", clear_btn_start)
        # print("clear_btn_end", clear_btn_end)

        if x_index_finger > clear_btn_start[0] and y_index_finger > clear_btn_start[1]:
            self._selected_nodes = []
            self._selected_nodes_id = []
        


    
    def handle_stream(self):
        cap = self._read_stream(-1)
        
        nodes = self._calc_nodes(self._video_width, self._video_height)        

        with self._mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while cap.isOpened():

                
                #  Check authentication
                is_authorized = False                
                if  self._selected_nodes_id == self._gt_path:
                    is_authorized = True
                    

                success, image = cap.read()
                image = cv2.resize(image, (self._video_width, self._video_height))



                # x_offset=y_offset=50
                # image[y_offset:y_offset+self._check_image.shape[0], x_offset:x_offset+self._check_image.shape[1]] = self._check_image
                
                if not(is_authorized):
                    self._make_gt_visible(image, nodes)

                
                if is_authorized:
                    cv2.putText(image, "Authorized", (image_width//2 -50, image_height//2-50), cv2.FONT_HERSHEY_COMPLEX, 1, (120, 110, 225), 2)

                
                # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                if not success:
                    continue

                results = hands.process(image)
                
                image_width = image.shape[1]
                image_height = image.shape[0]

                if not(is_authorized):
                    self._draw_nodes(image, nodes)
                    clear_btn_start, clear_btn_end  = self._draw_clear_btn(image, image_width, image_height)
            
                    for node, _ in self._selected_nodes:
                        cv2.circle(image, node, 20, (10, 100, 200), -1)
                

                
                    if len(self._selected_nodes) >= 2:
                        for index, (s, _) in enumerate(self._selected_nodes):
                            try:
                                cv2.line(image, s, self._selected_nodes[index+1][0], (200, 100, 150), 2)    
                            except:
                                continue

                    if len(self._selected_nodes) > 0:
                        cv2.line(image, self._selected_nodes[-1][0], self._last_index_finger_coord, (200, 100, 150), 2)


                
                image.flags.writeable = False
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # print(hand_landmarks)
                        self._draw_landmarks(image, hand_landmarks)

                        x_index_finger = int(hand_landmarks.landmark[self._mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width)
                        y_index_finger = int(hand_landmarks.landmark[self._mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_height)

                        self._check_clearance(x_index_finger, y_index_finger, clear_btn_start, clear_btn_end)

                        # cv2.circle(image, clear_btn_start, 10, color=(60, 98, 220), thickness=-1)
                        # cv2.circle(image, clear_btn_end, 10, color=(120, 129, 129), thickness=-1)

                        
                        self._last_index_finger_coord = (x_index_finger, y_index_finger)

                        if not(is_authorized):
                            self._set_path(nodes, x_index_finger, y_index_finger)
                        


                if not(is_authorized):
                    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
                else:
                    cv2.imshow('MediaPipe Hands', self._check_image)

                    # time.sleep(6)
                    # cap.release()        

                if cv2.waitKey(5) & 0xFF == 27:
                    break
            cap.release()


if __name__ == "__main__":
    pr = PatternRecognizer()
    pr.handle_stream()
